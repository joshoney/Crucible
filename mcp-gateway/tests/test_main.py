import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from fastmcp.exceptions import ToolError
from temporalio.client import WorkflowExecutionStatus

# Import the tools directly from your FastMCP setup
from main import (
    test_model as run_test_model,
    get_evaluation_status,
    get_evaluation_results,
    cancel_evaluation,
)

@pytest.fixture
def mock_temporal_client():
    """Fixture to mock the Temporal Client so tests run instantly without infrastructure."""
    with patch("main.get_temporal_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get_workflow_handle = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client

def make_handle(mock_temporal_client, status):
    mock_handle = AsyncMock()
    mock_description = MagicMock()
    mock_description.status = status
    mock_handle.describe.return_value = mock_description
    mock_temporal_client.get_workflow_handle.return_value = mock_handle
    return mock_handle

@pytest.mark.asyncio
async def test_submit_model_evaluation(mock_temporal_client):
    """Test that submitting a model generates a valid taskID and starts the workflow."""
    response = await run_test_model(model_name="llama-3-8b-instruct", config={"ctx_size": 4096})

    assert response["status"] == "IN_PROGRESS"
    assert response["task_id"].startswith("eval-llama-3-8b-instruct-")

    mock_temporal_client.start_workflow.assert_called_once()
    call_args, call_kwargs = mock_temporal_client.start_workflow.call_args
    assert call_args[0] == "LlmEvaluationWorkflow"  # Workflow name
    assert call_args[1] == {                        # Payload
        "model_name": "llama-3-8b-instruct",
        "config": {"ctx_size": 4096}
    }
    assert call_kwargs["task_queue"] == "eval-task-queue"
    assert call_kwargs["id"] == response["task_id"]

@pytest.mark.asyncio
async def test_submit_model_evaluation_no_config(mock_temporal_client):
    """Test that submitting a model without a config defaults to an empty dict."""
    await run_test_model(model_name="llama-3")
    call_args, _ = mock_temporal_client.start_workflow.call_args
    assert call_args[1] == {"model_name": "llama-3", "config": {}}

@pytest.mark.asyncio
async def test_submit_model_evaluation_with_slashes(mock_temporal_client):
    """Test that submitting a model name with slashes replaces them with dashes in taskID."""
    response = await run_test_model(model_name="meta-llama/Llama-3-8b")
    assert response["task_id"].startswith("eval-meta-llama-Llama-3-8b-")

@pytest.mark.asyncio
async def test_submit_model_evaluation_temporal_down(mock_temporal_client):
    mock_temporal_client.start_workflow.side_effect = RuntimeError("connection refused")
    with pytest.raises(ToolError, match="connection refused"):
        await run_test_model(model_name="llama-3")

@pytest.mark.asyncio
@pytest.mark.parametrize("temporal_status, expected", [
    (WorkflowExecutionStatus.RUNNING, "IN_PROGRESS"),
    (WorkflowExecutionStatus.COMPLETED, "COMPLETED"),
    (WorkflowExecutionStatus.FAILED, "FAILED"),
    (WorkflowExecutionStatus.TERMINATED, "FAILED"),
    (WorkflowExecutionStatus.TIMED_OUT, "FAILED"),
    (WorkflowExecutionStatus.CANCELED, "CANCELLED"),
    (WorkflowExecutionStatus.CONTINUED_AS_NEW, "UNKNOWN"),
])
async def test_get_evaluation_status_mapping(mock_temporal_client, temporal_status, expected):
    make_handle(mock_temporal_client, temporal_status)
    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"

    result = await get_evaluation_status(task_id=task_id)

    assert result == {"task_id": task_id, "status": expected, "temporal_status": temporal_status.name}
    mock_temporal_client.get_workflow_handle.assert_called_once_with(task_id)

@pytest.mark.asyncio
async def test_get_evaluation_status_error_raises_tool_error(mock_temporal_client):
    """If Temporal throws (e.g., taskID not found), the tool raises a ToolError."""
    mock_temporal_client.get_workflow_handle.side_effect = Exception("Workflow not found")
    with pytest.raises(ToolError, match="Workflow not found"):
        await get_evaluation_status(task_id="eval-invalid-123")

@pytest.mark.asyncio
async def test_get_evaluation_results_completed(mock_temporal_client):
    handle = make_handle(mock_temporal_client, WorkflowExecutionStatus.COMPLETED)
    handle.result.return_value = {"status": "completed", "file_count": 2}

    result = await get_evaluation_results(task_id="eval-1")

    assert result == {"task_id": "eval-1", "status": "COMPLETED", "result": {"status": "completed", "file_count": 2}}
    handle.query.assert_not_called()

@pytest.mark.asyncio
async def test_get_evaluation_results_cancelled_uses_summary_query(mock_temporal_client):
    handle = make_handle(mock_temporal_client, WorkflowExecutionStatus.CANCELED)
    handle.query.return_value = {"status": "cancelled", "partial": True}

    result = await get_evaluation_results(task_id="eval-1")

    assert result["status"] == "CANCELLED"
    assert result["result"]["partial"] is True
    handle.query.assert_called_once_with("summary")

@pytest.mark.asyncio
async def test_get_evaluation_results_unavailable_raises_tool_error(mock_temporal_client):
    handle = make_handle(mock_temporal_client, WorkflowExecutionStatus.FAILED)
    handle.query.side_effect = RuntimeError("no worker")
    with pytest.raises(ToolError, match="No results available"):
        await get_evaluation_results(task_id="eval-1")

@pytest.mark.asyncio
async def test_cancel_evaluation(mock_temporal_client):
    handle = make_handle(mock_temporal_client, WorkflowExecutionStatus.RUNNING)

    result = await cancel_evaluation(task_id="eval-1")

    assert result == {"task_id": "eval-1", "status": "CANCEL_REQUESTED"}
    handle.cancel.assert_awaited_once()

@pytest.mark.asyncio
async def test_cancel_evaluation_not_running(mock_temporal_client):
    handle = make_handle(mock_temporal_client, WorkflowExecutionStatus.COMPLETED)
    with pytest.raises(ToolError, match="not running"):
        await cancel_evaluation(task_id="eval-1")
    handle.cancel.assert_not_called()
