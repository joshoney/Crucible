import os
import uuid
from typing import Optional, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from temporalio.client import Client, WorkflowExecutionStatus

# 1. Initialize FastMCP
mcp = FastMCP(name="Crucible")

TEMPORAL_URL = os.getenv("TEMPORAL_URL", "temporal:7233")
TASK_QUEUE = "eval-task-queue"

# Temporal's native states mapped to API-friendly ones
STATUS_MAP = {
    WorkflowExecutionStatus.RUNNING: "IN_PROGRESS",
    WorkflowExecutionStatus.COMPLETED: "COMPLETED",
    WorkflowExecutionStatus.FAILED: "FAILED",
    WorkflowExecutionStatus.TERMINATED: "FAILED",
    WorkflowExecutionStatus.TIMED_OUT: "FAILED",
    WorkflowExecutionStatus.CANCELED: "CANCELLED",
}

# Cache the Temporal client to reuse the gRPC connection
_temporal_client: Optional[Client] = None

async def get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(TEMPORAL_URL)
    return _temporal_client

async def _describe(task_id: str):
    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(task_id)
        description = await handle.describe()
    except Exception as e:
        raise ToolError(f"Error retrieving status for {task_id}: {e}") from e
    return handle, STATUS_MAP.get(description.status, "UNKNOWN"), description.status

# 2. Expose the Submission Tool
@mcp.tool
async def test_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Submits an LLM to the platform for evaluation.
    Returns {"task_id", "status"}; the task_id MUST be used to poll the status.
    """
    client = await get_temporal_client()

    # Generate a unique task ID that safely handles model strings with slashes
    safe_name = model_name.replace("/", "-")
    task_id = f"eval-{safe_name}-{uuid.uuid4().hex[:8]}"

    payload = {
        "model_name": model_name,
        "config": config or {}
    }

    # Start the workflow asynchronously.
    # Passing the name as a string decouples this gateway from the worker codebase.
    try:
        await client.start_workflow(
            "LlmEvaluationWorkflow",
            payload,
            id=task_id,
            task_queue=TASK_QUEUE
        )
    except Exception as e:
        raise ToolError(f"Could not start evaluation for {model_name}: {e}") from e

    return {"task_id": task_id, "status": "IN_PROGRESS"}

# 3. Expose the Status Polling Tool
@mcp.tool
async def get_evaluation_status(task_id: str) -> Dict[str, str]:
    """
    Checks the status of a model evaluation task using the taskID.
    Returns {"task_id", "status"}: IN_PROGRESS, COMPLETED, FAILED, CANCELLED, or UNKNOWN.
    """
    _, status, temporal_status = await _describe(task_id)
    return {"task_id": task_id, "status": status, "temporal_status": temporal_status.name}

# 4. Results of a finished (or in-flight) evaluation
@mcp.tool
async def get_evaluation_results(task_id: str) -> Dict[str, Any]:
    """
    Returns the evaluation's summary: model, attempt, published files and commit.
    For a COMPLETED task this is the workflow result. For other states it is the workflow's
    live summary (e.g. which completed runs a CANCELLED task published as partial results).
    """
    handle, status, _ = await _describe(task_id)
    try:
        if status == "COMPLETED":
            result = await handle.result()
        else:
            result = await handle.query("summary")
    except Exception as e:
        raise ToolError(f"No results available for {task_id} ({status}): {e}") from e
    return {"task_id": task_id, "status": status, "result": result}

# 5. Safe mid-run cancellation
@mcp.tool
async def cancel_evaluation(task_id: str) -> Dict[str, str]:
    """
    Requests cancellation of a running evaluation. The worker stops evaluerBench gracefully,
    publishes any runs that already completed as a partial result, and the task ends CANCELLED.
    """
    handle, status, _ = await _describe(task_id)
    if status != "IN_PROGRESS":
        raise ToolError(f"Evaluation {task_id} is not running (status: {status})")
    try:
        await handle.cancel()
    except Exception as e:
        raise ToolError(f"Could not cancel {task_id}: {e}") from e
    return {"task_id": task_id, "status": "CANCEL_REQUESTED"}

# The app object is picked up by Uvicorn in the Dockerfile
app = mcp.http_app(transport="sse")
