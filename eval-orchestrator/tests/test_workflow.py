import pytest
import uuid
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio import activity
from temporalio.exceptions import ActivityError, ApplicationError
from workflow import LlmEvaluationWorkflow

@pytest.mark.asyncio
async def test_llm_evaluation_workflow():
    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    
    # Mock activities
    @activity.defn(name="provision_model")
    async def mock_provision_model(payload: dict) -> str:
        return "mock_provision"

    @activity.defn(name="run_agentic_evaluation")
    async def mock_run_agentic_evaluation(task_id: str, model_name: str) -> str:
        return "mock_eval"

    @activity.defn(name="publish_results")
    async def mock_publish_results(task_id: str) -> str:
        return "mock_publish"
        
    env = await WorkflowEnvironment.start_time_skipping()
    async with Worker(
            env.client,
            task_queue="eval-task-queue",
            workflows=[LlmEvaluationWorkflow],
            activities=[mock_provision_model, mock_run_agentic_evaluation, mock_publish_results],
        ):
            result = await env.client.execute_workflow(
                LlmEvaluationWorkflow.run,
                {"model_name": "llama-3"},
                id=task_id,
                task_queue="eval-task-queue",
            )
            assert result == "Successfully evaluated and published llama-3."

@pytest.mark.asyncio
async def test_llm_evaluation_workflow_failure():
    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    
    @activity.defn(name="provision_model")
    async def mock_provision_model_fail(payload: dict) -> str:
        raise ApplicationError("Simulated failure")

    @activity.defn(name="run_agentic_evaluation")
    async def mock_run_agentic_evaluation(task_id: str, model_name: str) -> str:
        return "mock_eval"

    @activity.defn(name="publish_results")
    async def mock_publish_results(task_id: str) -> str:
        return "mock_publish"
        
    env = await WorkflowEnvironment.start_time_skipping()
    async with Worker(
            env.client,
            task_queue="eval-task-queue",
            workflows=[LlmEvaluationWorkflow],
            activities=[mock_provision_model_fail, mock_run_agentic_evaluation, mock_publish_results],
        ):
            from temporalio.client import WorkflowFailureError
            with pytest.raises(WorkflowFailureError) as exc_info:
                await env.client.execute_workflow(
                    LlmEvaluationWorkflow.run,
                    {"model_name": "llama-3"},
                    id=task_id,
                    task_queue="eval-task-queue",
                )
            
            assert isinstance(exc_info.value.cause, ActivityError)
            assert "Simulated failure" in str(exc_info.value.cause.cause)
