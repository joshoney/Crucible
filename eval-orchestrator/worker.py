import asyncio
import concurrent.futures
import os
from datetime import timedelta
from temporalio.client import Client
from temporalio.worker import Worker
from activities import cleanup_scratch, load_latest_manifest, publish_results, record_outcome, run_evaluation
from workflow import LlmEvaluationWorkflow

TEMPORAL_URL = os.getenv("TEMPORAL_URL", "temporal:7233")
TASK_QUEUE = "eval-task-queue"
ACTIVITIES = [run_evaluation, publish_results, load_latest_manifest, record_outcome, cleanup_scratch]

async def main():
    client = await Client.connect(TEMPORAL_URL)

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LlmEvaluationWorkflow],
        activities=ACTIVITIES,
        activity_executor=concurrent.futures.ThreadPoolExecutor(max_workers=10),
        # One activity at a time: the Lemonade host has a single GPU
        max_concurrent_activities=1,
        # Heartbeats carry cancellation back to sync activities; flush them at least every 10s
        # (default 60s) so a cancel reaches evaluerBench quickly
        max_heartbeat_throttle_interval=timedelta(seconds=10),
    )

    print(f"Crucible worker started. Polling Temporal on {TEMPORAL_URL}...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
