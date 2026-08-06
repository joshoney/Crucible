import os
import io
import json
import requests
from temporalio import activity
from minio import Minio
from minio.error import S3Error
from github import Github, Auth, InputGitTreeElement
from typing import Dict, Any

# Environment Variables
LEMONADE_API_URL = os.getenv("LEMONADE_API_URL", "http://192.168.0.51:8000/api/v1")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "minio:9000").replace("http://", "").replace("https://", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "eval-artifacts")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. "joshoney/portfolio"
TARGET_BRANCH = os.getenv("TARGET_BRANCH", "main")

def get_minio_client() -> Minio:
    return Minio(
        S3_ENDPOINT,
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
        secure=False
    )

@activity.defn
async def provision_model(payload: Dict[str, Any]) -> str:
    """Instructs the Lemonade server on the local LAN to load the model."""
    model_name = payload["model_name"]
    config = payload.get("config", {})
    request_data = {"model_name": model_name, **config}
    
    activity.logger.info(f"Provisioning {model_name} at {LEMONADE_API_URL}/load")
    response = requests.post(f"{LEMONADE_API_URL}/load", json=request_data, timeout=120)
    response.raise_for_status()
    
    return f"{model_name} successfully provisioned."

@activity.defn
async def run_agentic_evaluation(task_id: str, model_name: str) -> str:
    """
    Executes evaluerBench, writes output locally, and offloads heavy artifacts to MinIO.
    This pattern keeps the Temporal workflow state history lean.
    """
    from evaluerBench.main import run_evaluation_suite # Imported from mounted volume
    
    scratchpad_dir = f"/app/scratchpad/{task_id}"
    os.makedirs(scratchpad_dir, exist_ok=True)
    
    activity.logger.info(f"Running evaluerBench for model {model_name}...")
    run_evaluation_suite(model=model_name, output_dir=scratchpad_dir)
    
    minio_client = get_minio_client()
    
    # Ensure bucket exists
    if not minio_client.bucket_exists(S3_BUCKET):
        minio_client.make_bucket(S3_BUCKET)
        
    uploaded_files = 0
    for root, _, files in os.walk(scratchpad_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Store artifacts scoped by task_id
            s3_path = f"{task_id}/{file}"
            
            minio_client.fput_object(S3_BUCKET, s3_path, local_path)
            uploaded_files += 1
            activity.logger.info(f"Uploaded {file} to MinIO as {s3_path}")
            
    return f"Uploaded {uploaded_files} files to s3://{S3_BUCKET}/{task_id}"

@activity.defn
async def publish_results(task_id: str) -> str:
    """Pulls artifacts from MinIO and pushes an atomic commit to Astro via GitHub REST API."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise ValueError("Missing GITHUB_TOKEN or GITHUB_REPO environment variables.")

    minio_client = get_minio_client()
    auth = Auth.Token(GITHUB_TOKEN)
    gh_client = Github(auth=auth)
    
    repo = gh_client.get_repo(GITHUB_REPO)
    ref = repo.get_git_ref(f"heads/{TARGET_BRANCH}")
    base_commit = repo.get_git_commit(ref.object.sha)
    base_tree = repo.get_git_tree(base_commit.tree.sha)
    
    tree_elements = []
    
    # List and retrieve objects for this task ID from MinIO
    objects = minio_client.list_objects(S3_BUCKET, prefix=f"{task_id}/")
    for obj in objects:
        response = minio_client.get_object(S3_BUCKET, obj.object_name)
        file_bytes = response.read()
        file_content = file_bytes.decode('utf-8')
        response.close()
        response.release_conn()
        
        filename = obj.object_name.replace(f"{task_id}/", "")
        
        # Determine target path in the remote Astro repository
        target_path = f"src/resources/evaluerBench/{task_id}/{filename}"
            
        tree_elements.append(
            InputGitTreeElement(
                path=target_path,
                mode='100644',
                type='blob',
                content=file_content
            )
        )
        
    if not tree_elements:
        raise RuntimeError(f"No artifacts found in MinIO for task_id: {task_id}")

    # Atomic Commit via Git Data API
    new_tree = repo.create_git_tree(tree_elements, base_tree)
    commit_msg = f"Agent: Auto-publish eval results for task {task_id}"
    new_commit = repo.create_git_commit(commit_msg, new_tree, [base_commit])
    ref.edit(new_commit.sha)
    
    activity.logger.info(f"Pushed atomic commit {new_commit.sha} to {GITHUB_REPO}")
    return f"Successfully published task {task_id} to GitHub repo {GITHUB_REPO}."