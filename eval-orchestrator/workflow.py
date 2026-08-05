from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

# Safe import for activities to comply with Temporal's determinism constraints
with workflow.unsafe.imports_passed_through():
    from activities import (
        provision_model,
        run_agentic_evaluation,
        publish_results,
    )

@workflow.defn
class LlmEvaluationWorkflow:
    @workflow.run
    async def run(self, payload: dict) -> str:
        model_name = payload["model_name"]
        task_id = workflow.info().workflow_id
        
        standard_retry = RetryPolicy(
            initial_interval=timedelta(seconds=5),
            maximum_interval=timedelta(minutes=1),
            maximum_attempts=3,
        )

        # 1. Provision the model on local hardware
        await workflow.execute_activity(
            provision_model,
            payload,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=standard_retry,
        )
        
        # 2. Run the LangGraph evaluation and push to S3 bucket
        await workflow.execute_activity(
            run_agentic_evaluation,
            args=[task_id, model_name],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=standard_retry,
        )
        
        # 3. Publish results directly to Astro repository
        await workflow.execute_activity(
            publish_results,
            args=[task_id],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=standard_retry,
        )
        
        return f"Successfully evaluated and published {model_name}."