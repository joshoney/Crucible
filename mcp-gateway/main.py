import os
import uuid
from typing import Optional, Dict, Any
from fastmcp import FastMCP
from temporalio.client import Client, WorkflowExecutionStatus

# 1. Initialize FastMCP
mcp = FastMCP(name="EvalPlatformGateway")

TEMPORAL_URL = os.getenv("TEMPORAL_URL", "temporal:7233")
TASK_QUEUE = "eval-task-queue"

# Cache the Temporal client to reuse the gRPC connection
_temporal_client: Optional[Client] = None

async def get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        _temporal_client = await Client.connect(TEMPORAL_URL)
    return _temporal_client

# 2. Expose the Submission Tool
@mcp.tool
async def test_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Submits an LLM to the platform for evaluation. 
    Returns a taskID that MUST be used to poll the status.
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
    await client.start_workflow(
        "LlmEvaluationWorkflow",
        payload,
        id=task_id,
        task_queue=TASK_QUEUE
    )
    
    return f"Evaluation started. taskID: {task_id}"

# 3. Expose the Status Polling Tool
@mcp.tool
async def get_evaluation_status(task_id: str) -> str:
    """
    Checks the status of a model evaluation task using the taskID.
    Returns IN_PROGRESS, COMPLETED, or FAILED.
    """
    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(task_id)
        description = await handle.describe()
        
        status = description.status
        
        # Map Temporal's native states to clean, API-friendly strings
        if status == WorkflowExecutionStatus.RUNNING:
            return "IN_PROGRESS"
        elif status == WorkflowExecutionStatus.COMPLETED:
            return "COMPLETED"
        elif status in (
            WorkflowExecutionStatus.FAILED, 
            WorkflowExecutionStatus.CANCELED, 
            WorkflowExecutionStatus.TERMINATED
        ):
            return "FAILED"
        else:
            return f"UNKNOWN_STATE: {status.name}"
            
    except Exception as e:
        return f"Error retrieving status for {task_id}: {str(e)}"
# The app object is picked up by Uvicorn in the Dockerfile
app = mcp.http_app(transport="sse")