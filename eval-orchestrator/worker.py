import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker
from activities import provision_model, run_agentic_evaluation, publish_results
from workflow import LlmEvaluationWorkflow

TEMPORAL_URL = os.getenv("TEMPORAL_URL", "temporal:7233")
TASK_QUEUE = "eval-task-queue"

async def main():
    client = await Client.connect(TEMPORAL_URL)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LlmEvaluationWorkflow],
        activities=[provision_model, run_agentic_evaluation, publish_results],
        # Protects your Lemonade hardware by strictly limiting concurrency
        max_concurrent_activities=1, 
    )
    
    print(f"Eval Platform Worker started. Polling Temporal on {TEMPORAL_URL}...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())