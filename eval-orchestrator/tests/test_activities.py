import pytest
import sys
import requests
from unittest.mock import patch, MagicMock

# Mock evaluerBench before importing activities
mock_evaluer_bench = MagicMock()
sys.modules['evaluerBench'] = mock_evaluer_bench
sys.modules['evaluerBench.main'] = mock_evaluer_bench

from activities import provision_model, run_agentic_evaluation, publish_results
from temporalio.testing import ActivityEnvironment

@pytest.mark.asyncio
async def test_provision_model():
    env = ActivityEnvironment()
    with patch("activities.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        result = await env.run(provision_model, {"model_name": "llama-3"})
        assert result == "llama-3 successfully provisioned."
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_run_agentic_evaluation():
    env = ActivityEnvironment()
    with patch("activities.get_minio_client") as mock_get_minio, \
         patch("activities.os.walk") as mock_walk:
         
        mock_minio = MagicMock()
        mock_minio.bucket_exists.return_value = False
        mock_get_minio.return_value = mock_minio
        
        mock_walk.return_value = [("/app/scratchpad/test-task", [], ["bench.json"])]
        
        result = await env.run(run_agentic_evaluation, "test-task", "llama-3")
        
        assert "Uploaded 1 files to s3" in result
        mock_evaluer_bench.run_evaluation_suite.assert_called_once_with(model="llama-3", output_dir="/app/scratchpad/test-task")
        mock_minio.make_bucket.assert_called_once()
        mock_minio.fput_object.assert_called_once()

@pytest.mark.asyncio
async def test_publish_results():
    env = ActivityEnvironment()
    with patch("activities.get_minio_client") as mock_get_minio, \
         patch("activities.Github") as mock_github, \
         patch("activities.GITHUB_TOKEN", "fake_token"), \
         patch("activities.GITHUB_REPO", "fake/repo"):
        
        mock_minio = MagicMock()
        mock_get_minio.return_value = mock_minio
        mock_obj = MagicMock()
        mock_obj.object_name = "test-task/bench.json"
        mock_minio.list_objects.return_value = [mock_obj]
        
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"hello": "world"}'
        mock_minio.get_object.return_value = mock_response
        
        mock_gh_instance = MagicMock()
        mock_github.return_value = mock_gh_instance
        
        result = await env.run(publish_results, "test-task")
        
        assert "Successfully published task test-task" in result
        mock_gh_instance.get_repo.assert_called_once()

@pytest.mark.asyncio
async def test_provision_model_http_error():
    env = ActivityEnvironment()
    with patch("activities.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad Request")
        mock_post.return_value = mock_response
        
        with pytest.raises(requests.exceptions.HTTPError):
            await env.run(provision_model, {"model_name": "llama-3"})

@pytest.mark.asyncio
async def test_run_agentic_evaluation_bucket_exists():
    env = ActivityEnvironment()
    with patch("activities.get_minio_client") as mock_get_minio, \
         patch("activities.os.walk") as mock_walk:
         
        mock_minio = MagicMock()
        mock_minio.bucket_exists.return_value = True
        mock_get_minio.return_value = mock_minio
        
        mock_walk.return_value = [("/app/scratchpad/test-task", [], ["bench.json"])]
        
        result = await env.run(run_agentic_evaluation, "test-task", "llama-3")
        
        mock_minio.make_bucket.assert_not_called()
        mock_minio.fput_object.assert_called_once()

@pytest.mark.asyncio
async def test_publish_results_missing_env():
    env = ActivityEnvironment()
    with patch("activities.GITHUB_TOKEN", None), \
         patch("activities.GITHUB_REPO", None):
        with pytest.raises(ValueError, match="Missing GITHUB_TOKEN or GITHUB_REPO environment variables"):
            await env.run(publish_results, "test-task")

@pytest.mark.asyncio
async def test_publish_results_file_routing():
    env = ActivityEnvironment()
    with patch("activities.get_minio_client") as mock_get_minio, \
         patch("activities.Github") as mock_github, \
         patch("activities.GITHUB_TOKEN", "fake_token"), \
         patch("activities.GITHUB_REPO", "fake/repo"):
        
        mock_minio = MagicMock()
        mock_get_minio.return_value = mock_minio
        
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "test-task/bench.json"
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "test-task/log.txt"
        
        mock_minio.list_objects.return_value = [mock_obj1, mock_obj2]
        
        mock_response = MagicMock()
        mock_response.read.return_value = b'content'
        mock_minio.get_object.return_value = mock_response
        
        mock_gh_instance = MagicMock()
        mock_github.return_value = mock_gh_instance
        
        with patch("activities.InputGitTreeElement") as mock_element:
            await env.run(publish_results, "test-task")
            
            assert mock_element.call_count == 2
            args_list = mock_element.call_args_list
            paths = [kwargs.get('path') for args, kwargs in args_list]
            assert "results/test-task/bench.json" in paths
            assert "public/artifacts/test-task/log.txt" in paths

@pytest.mark.asyncio
async def test_publish_results_empty_artifacts():
    env = ActivityEnvironment()
    with patch("activities.get_minio_client") as mock_get_minio, \
         patch("activities.Github"), \
         patch("activities.GITHUB_TOKEN", "fake_token"), \
         patch("activities.GITHUB_REPO", "fake/repo"):
        
        mock_minio = MagicMock()
        mock_minio.list_objects.return_value = []
        mock_get_minio.return_value = mock_minio
        
        with pytest.raises(RuntimeError, match="No artifacts found in MinIO for task_id: test-task"):
            await env.run(publish_results, "test-task")
