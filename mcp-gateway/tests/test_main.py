import pytest
import uuid
from unittest.mock import AsyncMock, patch
from temporalio.client import WorkflowExecutionStatus

# Import the tools directly from your FastMCP setup
from main import test_model as run_test_model, get_evaluation_status
from unittest.mock import MagicMock

@pytest.fixture
def mock_temporal_client():
    """Fixture to mock the Temporal Client so tests run instantly without infrastructure."""
    with patch("main.get_temporal_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get_workflow_handle = MagicMock()
        mock_get_client.return_value = mock_client
        yield mock_client

@pytest.mark.asyncio
async def test_submit_model_evaluation(mock_temporal_client):
    """Test that submitting a model generates a valid taskID and starts the workflow."""
    
    # Act
    response = await run_test_model(model_name="llama-3-8b-instruct", config={"ctx_size": 4096})
    
    # Assert
    assert "Evaluation started. taskID: eval-llama-3-8b-instruct-" in response
    
    # Verify the Temporal Client was called with the correct parameters
    mock_temporal_client.start_workflow.assert_called_once()
    
    # Inspect the arguments passed to start_workflow
    call_args, call_kwargs = mock_temporal_client.start_workflow.call_args
    assert call_args[0] == "LlmEvaluationWorkflow"  # Workflow name
    assert call_args[1] == {                        # Payload
        "model_name": "llama-3-8b-instruct", 
        "config": {"ctx_size": 4096}
    }
    assert call_kwargs["task_queue"] == "eval-task-queue"
    assert "eval-llama-3-8b-instruct-" in call_kwargs["id"]

@pytest.mark.asyncio
async def test_submit_model_evaluation_no_config(mock_temporal_client):
    """Test that submitting a model without a config defaults to an empty dict."""
    response = await run_test_model(model_name="llama-3")
    assert "Evaluation started" in response
    
    call_args, _ = mock_temporal_client.start_workflow.call_args
    assert call_args[1] == {
        "model_name": "llama-3",
        "config": {}
    }

@pytest.mark.asyncio
async def test_submit_model_evaluation_with_slashes(mock_temporal_client):
    """Test that submitting a model name with slashes replaces them with dashes in taskID."""
    response = await run_test_model(model_name="meta-llama/Llama-3-8b")
    
    assert "eval-meta-llama-Llama-3-8b-" in response
    _, call_kwargs = mock_temporal_client.start_workflow.call_args
    assert "eval-meta-llama-Llama-3-8b-" in call_kwargs["id"]

@pytest.mark.asyncio
async def test_get_evaluation_status_running(mock_temporal_client):
    """Test that a RUNNING Temporal workflow maps to IN_PROGRESS."""
    mock_handle = AsyncMock()
    mock_description = AsyncMock()
    mock_description.status = WorkflowExecutionStatus.RUNNING
    mock_handle.describe.return_value = mock_description
    mock_temporal_client.get_workflow_handle.return_value = mock_handle
    
    status = await get_evaluation_status(task_id="eval-123")
    assert status == "IN_PROGRESS"

@pytest.mark.asyncio
async def test_get_evaluation_status_completed(mock_temporal_client):
    """Test that a COMPLETED Temporal workflow maps to the correct string."""
    
    # Setup the mock to return a COMPLETED status
    mock_handle = AsyncMock()
    mock_description = AsyncMock()
    mock_description.status = WorkflowExecutionStatus.COMPLETED
    mock_handle.describe.return_value = mock_description
    mock_temporal_client.get_workflow_handle.return_value = mock_handle
    
    # Act
    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    status = await get_evaluation_status(task_id=task_id)
    
    # Assert
    assert status == "COMPLETED"
    mock_temporal_client.get_workflow_handle.assert_called_once_with(task_id)

@pytest.mark.asyncio
@pytest.mark.parametrize("temporal_status", [
    WorkflowExecutionStatus.FAILED,
    WorkflowExecutionStatus.CANCELED,
    WorkflowExecutionStatus.TERMINATED
])
async def test_get_evaluation_status_failed_states(mock_temporal_client, temporal_status):
    """Test that FAILED, CANCELED, and TERMINATED map to FAILED."""
    mock_handle = AsyncMock()
    mock_description = AsyncMock()
    mock_description.status = temporal_status
    mock_handle.describe.return_value = mock_description
    mock_temporal_client.get_workflow_handle.return_value = mock_handle
    
    status = await get_evaluation_status(task_id="eval-failed-123")
    assert status == "FAILED"

@pytest.mark.asyncio
async def test_get_evaluation_status_unknown(mock_temporal_client):
    """Test that unhandled states map to UNKNOWN_STATE."""
    mock_handle = AsyncMock()
    mock_description = AsyncMock()
    mock_description.status = WorkflowExecutionStatus.CONTINUED_AS_NEW
    mock_handle.describe.return_value = mock_description
    mock_temporal_client.get_workflow_handle.return_value = mock_handle
    
    status = await get_evaluation_status(task_id="eval-unknown-123")
    assert status == "UNKNOWN_STATE: CONTINUED_AS_NEW"

@pytest.mark.asyncio
async def test_get_evaluation_status_error_handling(mock_temporal_client):
    """Test that if Temporal throws an error (e.g., taskID not found), it is handled cleanly."""
    
    # Force the mock client to throw an exception
    mock_temporal_client.get_workflow_handle.side_effect = Exception("Workflow not found")
    
    # Act
    status = await get_evaluation_status(task_id="eval-invalid-123")
    
    # Assert
    assert "Error retrieving status" in status
    assert "Workflow not found" in status