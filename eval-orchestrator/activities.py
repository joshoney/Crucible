import base64
import concurrent.futures
import contextvars
import hashlib
import importlib
import io
import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests
from github import Auth, Github, GithubException, InputGitTreeElement, UnknownObjectException
from minio import Minio
from temporalio import activity
from temporalio.exceptions import ApplicationError, CancelledError

# Environment Variables
LEMONADE_API_URL = os.getenv("LEMONADE_API_URL", "http://10.10.0.12:8000/api/v1")
# After /load, check GET {LEMONADE_API_URL}/health lists the model before the eval starts
LEMONADE_VERIFY_LOADED = os.getenv("LEMONADE_VERIFY_LOADED", "true").lower() not in ("0", "false", "no")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "minio:9000").replace("http://", "").replace("https://", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_BUCKET", "eval-artifacts")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g. "joshoney/devsite"
TARGET_BRANCH = os.getenv("TARGET_BRANCH", "main")
RESULTS_ROOT = "src/resources/evaluerBench"

# Per-task working directory for evaluerBench output (bind-mounted to the host in docker-compose)
SCRATCHPAD_ROOT = os.getenv("SCRATCHPAD_ROOT", "/app/scratchpad")
# Task scratch dirs older than this are deleted when an evaluation starts; <= 0 disables
SCRATCH_RETENTION_DAYS = float(os.getenv("SCRATCH_RETENTION_DAYS", "7"))

# How often evaluerBench calls on_tick (we heartbeat and check for cancellation each time)
POLL_INTERVAL_SECONDS = 5.0
# How long evaluerBench waits after SIGTERM before killing the CLI
CANCEL_GRACE_SECONDS = 30.0


class _NeverRaised(Exception):
    """Stand-in when the installed evaluerBench has no EvaluationCancelled."""


class _ProvisionCancelled(Exception):
    """Cancellation arrived before evaluerBench started."""


def get_minio_client() -> Minio:
    return Minio(
        S3_ENDPOINT,
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
        secure=False
    )


def _load_evaluerbench():
    # evaluerBench is git-cloned into /opt by the Dockerfile (PYTHONPATH=/opt); imported lazily
    # so this module can be imported (and unit-tested) without evaluerBench installed
    eb_main = importlib.import_module("evaluerBench.main")
    cancelled_exc = getattr(eb_main, "EvaluationCancelled", None)
    if not (isinstance(cancelled_exc, type) and issubclass(cancelled_exc, BaseException)):
        cancelled_exc = _NeverRaised
    return eb_main.run_evaluation_suite, cancelled_exc


def _task_dir(task_id: str) -> str:
    if not task_id or task_id in (".", "..") or "/" in task_id or "\\" in task_id:
        raise ApplicationError(f"Invalid task_id: {task_id!r}", non_retryable=True)
    return os.path.join(SCRATCHPAD_ROOT, task_id)


def git_blob_sha(data: bytes) -> str:
    """The SHA git (and GitHub) assigns to a blob with this content."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _in_activity_context(fn: Callable) -> Callable:
    """Bind fn to this activity's context, in case evaluerBench calls it from another thread."""
    ctx = contextvars.copy_context()

    def wrapper(*args):
        if activity.in_activity():
            return fn(*args)
        return ctx.copy().run(fn, *args)
    return wrapper


def _call_with_heartbeat(fn: Callable[[], Any], details: dict) -> Any:
    """Run a blocking call while heartbeating, so a slow /load can't trip the heartbeat timeout."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn)
        while True:
            try:
                return future.result(timeout=POLL_INTERVAL_SECONDS)
            except concurrent.futures.TimeoutError:
                activity.heartbeat(details)
                if activity.is_cancelled():
                    raise _ProvisionCancelled()
    finally:
        # Don't block on an abandoned request after cancellation
        pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Provisioning helpers (called by run_evaluation, inside the same activity attempt)
# ---------------------------------------------------------------------------

def provision_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> None:
    """Instructs the Lemonade server on the local LAN to load the model."""
    request_data = {"model_name": model_name, **(config or {})}
    activity.logger.info(f"Provisioning {model_name} at {LEMONADE_API_URL}/load")
    response = requests.post(f"{LEMONADE_API_URL}/load", json=request_data, timeout=120)
    response.raise_for_status()


def verify_model_loaded(model_name: str) -> None:
    """Fail the attempt (retryable) if Lemonade's /health doesn't list the model as loaded."""
    response = requests.get(f"{LEMONADE_API_URL}/health", timeout=30)
    response.raise_for_status()
    health = response.json()
    if "all_models_loaded" not in health and "model_loaded" not in health:
        activity.logger.warning("Lemonade /health has no loaded-model fields; skipping verification")
        return
    loaded = {m.get("model_name") for m in health.get("all_models_loaded") or [] if isinstance(m, dict)}
    loaded.add(health.get("model_loaded"))
    loaded.discard(None)
    if model_name not in loaded:
        raise ApplicationError(f"Lemonade reports {sorted(loaded)} loaded, not {model_name}", type="ModelNotLoaded")


# ---------------------------------------------------------------------------
# Scratch space
# ---------------------------------------------------------------------------

def cleanup_expired_scratch(root: str, retention_days: float, keep: Optional[str] = None) -> List[str]:
    """Delete task dirs under root older than retention_days. Only touches dirs that look
    like Crucible task dirs (they contain an attempt-<n> child), so a misconfigured root is safe."""
    if retention_days <= 0 or not os.path.isdir(root):
        return []
    cutoff = time.time() - retention_days * 86400
    removed = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name == keep or not os.path.isdir(path):
            continue
        try:
            if not any(child.startswith("attempt-") for child in os.listdir(path)):
                continue
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path)
                removed.append(name)
        except OSError as err:
            activity.logger.warning(f"Could not clean up scratch dir {path}: {err}")
    return removed


@activity.defn
def cleanup_scratch(task_id: str) -> bool:
    """Delete SCRATCHPAD_ROOT/<task_id> after a successful publish (the files are in MinIO and git)."""
    path = _task_dir(task_id)
    if not os.path.isdir(path):
        return False
    shutil.rmtree(path)
    return True


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _ensure_bucket(minio_client: Minio) -> None:
    if not minio_client.bucket_exists(S3_BUCKET):
        minio_client.make_bucket(S3_BUCKET)


def _put_json(minio_client: Minio, key: str, data: dict) -> None:
    body = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    minio_client.put_object(S3_BUCKET, key, io.BytesIO(body), len(body), content_type="application/json")


def _get_bytes(minio_client: Minio, key: str) -> bytes:
    response = minio_client.get_object(S3_BUCKET, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _upload_attempt(minio_client: Minio, task_id: str, attempt: int, attempt_dir: str) -> List[Dict[str, str]]:
    """Upload everything evaluerBench wrote in this attempt under <task_id>/attempt-<n>/."""
    files = []
    for root, _, names in os.walk(attempt_dir):
        for name in names:
            local_path = os.path.join(root, name)
            # rel_path follows evaluerBench's layout: <model>/<evalId>/<run file>
            rel_path = os.path.relpath(local_path, attempt_dir).replace("\\", "/")
            key = f"{task_id}/attempt-{attempt}/{rel_path}"
            minio_client.fput_object(S3_BUCKET, key, local_path)
            activity.heartbeat({"phase": "uploading", "attempt": attempt, "uploaded": len(files) + 1})
            files.append({"key": key, "rel_path": rel_path})
    files.sort(key=lambda f: f["rel_path"])
    return files


def _manifest_key(task_id: str, attempt: int) -> str:
    return f"{task_id}/manifests/attempt-{attempt}.json"


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

# no_thread_cancel_exception: cancellation is cooperative. Temporal must not inject an exception
# into this thread, because evaluerBench has to SIGTERM the CLI and let it save completed runs.
@activity.defn(no_thread_cancel_exception=True)
def run_evaluation(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Provisions the model on Lemonade, then runs evaluerBench, in ONE activity attempt, so no
    other workflow's provisioning can interleave. Each attempt writes to its own scratch dir
    and MinIO prefix, and returns a manifest of exactly what it uploaded.
    """
    run_evaluation_suite, evaluation_cancelled = _load_evaluerbench()
    model_name = payload["model_name"]
    info = activity.info()
    attempt = info.attempt

    if info.heartbeat_details:
        activity.logger.info(f"{task_id} attempt {attempt}; the previous attempt reached {info.heartbeat_details[0]}")

    cleanup_expired_scratch(SCRATCHPAD_ROOT, SCRATCH_RETENTION_DAYS, keep=task_id)
    attempt_dir = os.path.join(_task_dir(task_id), f"attempt-{attempt}")
    if os.path.exists(attempt_dir):
        shutil.rmtree(attempt_dir)
    os.makedirs(attempt_dir)

    progress: Dict[str, Any] = {}

    def on_tick(p: Dict[str, Any]) -> None:
        progress.update(p or {})
        activity.heartbeat({"phase": "evaluating", "attempt": attempt, **progress})

    status = "failed"
    error: Optional[BaseException] = None
    try:
        details = {"phase": "provisioning", "attempt": attempt}
        activity.heartbeat(details)
        _call_with_heartbeat(lambda: provision_model(model_name, payload.get("config")), details)
        if LEMONADE_VERIFY_LOADED:
            _call_with_heartbeat(lambda: verify_model_loaded(model_name), details)
        if activity.is_cancelled():
            raise _ProvisionCancelled()

        activity.logger.info(f"Running evaluerBench for model {model_name} (attempt {attempt})...")
        run_evaluation_suite(
            model=model_name,
            output_dir=attempt_dir,
            on_tick=_in_activity_context(on_tick),
            should_cancel=_in_activity_context(activity.is_cancelled),
            poll_interval=POLL_INTERVAL_SECONDS,
            grace_period=CANCEL_GRACE_SECONDS,
        )
        status = "completed"
    except (evaluation_cancelled, _ProvisionCancelled):
        status = "cancelled"
    except BaseException as err:
        error = err

    # Upload whatever this attempt produced. Failed attempts stay in MinIO for debugging but are
    # never published; a cancelled attempt's files are complete runs the CLI saved on SIGTERM.
    manifest: Dict[str, Any] = {}
    try:
        minio_client = get_minio_client()
        _ensure_bucket(minio_client)
        files = _upload_attempt(minio_client, task_id, attempt, attempt_dir)
        manifest = {
            "task_id": task_id,
            "model_name": model_name,
            "attempt": attempt,
            "status": status,
            "prefix": f"{task_id}/attempt-{attempt}/",
            "files": files,
            "file_count": len(files),
            "progress": progress,
        }
        _put_json(minio_client, _manifest_key(task_id, attempt), manifest)
        activity.logger.info(f"Attempt {attempt} {status}: uploaded {len(files)} files to s3://{S3_BUCKET}/{manifest['prefix']}")
    except Exception as upload_err:
        if error is None:
            raise
        # Don't let an upload problem hide the evaluation error
        activity.logger.warning(f"Upload after the failed attempt also failed: {upload_err}")

    if error is not None:
        raise error
    if status == "cancelled":
        # Completes the activity as cancelled; the workflow reads this attempt's manifest from MinIO
        raise CancelledError("Evaluation cancelled")
    return manifest


@activity.defn
def load_latest_manifest(task_id: str) -> Optional[Dict[str, Any]]:
    """Return the manifest of the highest attempt for task_id, or None if there is none."""
    minio_client = get_minio_client()
    prefix = f"{task_id}/manifests/"
    latest = None
    for obj in minio_client.list_objects(S3_BUCKET, prefix=prefix, recursive=True):
        name = obj.object_name[len(prefix):]
        if name.startswith("attempt-") and name.endswith(".json"):
            n = int(name[len("attempt-"):-len(".json")])
            if latest is None or n > latest[0]:
                latest = (n, obj.object_name)
    if latest is None:
        return None
    return json.loads(_get_bytes(minio_client, latest[1]))


@activity.defn
def record_outcome(task_id: str, record: Dict[str, Any]) -> str:
    """Write <task_id>/failure.json or <task_id>/cancelled.json to MinIO."""
    status = record.get("status", "outcome")
    key = f"{task_id}/{'failure' if status == 'failed' else status}.json"
    minio_client = get_minio_client()
    _ensure_bucket(minio_client)
    _put_json(minio_client, key, {"task_id": task_id, "recorded_at": datetime.now(timezone.utc).isoformat(), **record})
    return key


def _blob_sha_at(repo, path: str, ref_sha: str) -> Optional[str]:
    try:
        content = repo.get_contents(path, ref=ref_sha)
    except UnknownObjectException:
        return None
    return None if isinstance(content, list) else content.sha


@activity.defn
def publish_results(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Publishes exactly the manifest's files to GITHUB_REPO as one atomic commit (Git Data API).
    Idempotent: if HEAD already has identical blobs at every path, no commit is made."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        raise ValueError("Missing GITHUB_TOKEN or GITHUB_REPO environment variables.")
    task_id = manifest["task_id"]
    files = manifest.get("files") or []
    if not files:
        raise ApplicationError(f"Manifest for {task_id} has no files to publish", non_retryable=True)

    minio_client = get_minio_client()
    # Publish into the site's results tree using evaluerBench's layout (no task/attempt level),
    # so each run lands alongside earlier runs of the same model and eval
    contents = [(f"{RESULTS_ROOT}/{f['rel_path']}", _get_bytes(minio_client, f["key"])) for f in files]
    paths = [p for p, _ in contents]

    repo = Github(auth=Auth.Token(GITHUB_TOKEN)).get_repo(GITHUB_REPO)
    # Read HEAD on every attempt, so a retry after a conflict builds on the new tip
    ref = repo.get_git_ref(f"heads/{TARGET_BRANCH}")
    head_sha = ref.object.sha

    if all(_blob_sha_at(repo, path, head_sha) == git_blob_sha(data) for path, data in contents):
        activity.logger.info(f"{task_id}: all {len(contents)} files already at {head_sha}; not committing")
        return {"status": "already_published", "commit": head_sha, "paths": paths}

    base_commit = repo.get_git_commit(head_sha)
    base_tree = repo.get_git_tree(base_commit.tree.sha)
    tree_elements = []
    for target_path, data in contents:
        try:
            tree_elements.append(InputGitTreeElement(path=target_path, mode="100644", type="blob", content=data.decode("utf-8")))
        except UnicodeDecodeError:
            # Binary artifact (like an image): create a git blob first
            blob = repo.create_git_blob(base64.b64encode(data).decode("utf-8"), "base64")
            tree_elements.append(InputGitTreeElement(path=target_path, mode="100644", type="blob", sha=blob.sha))

    new_tree = repo.create_git_tree(tree_elements, base_tree)
    commit_msg = f"Agent: Auto-publish eval results for task {task_id} (attempt {manifest.get('attempt')})"
    if manifest.get("status") == "cancelled":
        commit_msg += " [partial: cancelled]"
    new_commit = repo.create_git_commit(commit_msg, new_tree, [base_commit])
    try:
        # Never force: if the branch moved since we read HEAD, fail and retry on top of it
        ref.edit(new_commit.sha, force=False)
    except GithubException as err:
        if err.status in (409, 422):
            raise ApplicationError(
                f"{TARGET_BRANCH} moved while publishing {task_id}; retrying on the new HEAD",
                type="PublishConflict",
            ) from err
        raise

    activity.logger.info(f"Pushed atomic commit {new_commit.sha} to {GITHUB_REPO}")
    return {"status": "published", "commit": new_commit.sha, "paths": paths}
