import asyncio
from datetime import timedelta
from typing import Any, Dict, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, is_cancelled_exception
from temporalio.workflow import ActivityCancellationType

# Safe import for activities to comply with Temporal's determinism constraints
with workflow.unsafe.imports_passed_through():
    from activities import (
        cleanup_scratch,
        load_latest_manifest,
        publish_results,
        record_outcome,
        run_evaluation,
    )

STANDARD_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=1),
    maximum_attempts=3,
)
# Ref-update conflicts are expected when something else commits to the branch; give them room
PUBLISH_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    maximum_interval=timedelta(minutes=1),
    maximum_attempts=5,
)
EVAL_TIMEOUT = timedelta(minutes=45)
EVAL_HEARTBEAT_TIMEOUT = timedelta(minutes=2)
SHORT_TIMEOUT = timedelta(minutes=2)


def _error_message(err: BaseException) -> str:
    # ActivityError wraps the activity's own exception; report that one
    cause = getattr(err, "cause", None)
    return str(cause or err)


@workflow.defn
class LlmEvaluationWorkflow:
    def __init__(self) -> None:
        self._summary: Dict[str, Any] = {"status": "running"}

    @workflow.query
    def summary(self) -> Dict[str, Any]:
        return self._summary

    @workflow.run
    async def run(self, payload: dict) -> Dict[str, Any]:
        model_name = payload["model_name"]
        task_id = workflow.info().workflow_id
        self._summary = {"task_id": task_id, "model_name": model_name, "status": "running"}
        manifest: Optional[Dict[str, Any]] = None

        try:
            # 1. Provision the model and run evaluerBench in ONE activity, then upload this
            #    attempt's artifacts to MinIO. Returns a manifest of exactly what was uploaded.
            manifest = await workflow.execute_activity(
                run_evaluation,
                args=[task_id, payload],
                start_to_close_timeout=EVAL_TIMEOUT,
                heartbeat_timeout=EVAL_HEARTBEAT_TIMEOUT,
                retry_policy=STANDARD_RETRY,
                # On workflow cancel, wait for the activity to stop the CLI and upload its files
                cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            )
            self._summary.update(status="publishing", attempt=manifest["attempt"], file_count=manifest["file_count"])

            # 2. Publish exactly the manifest's files to the site repo as one commit
            publish = await workflow.execute_activity(
                publish_results,
                manifest,
                start_to_close_timeout=SHORT_TIMEOUT,
                retry_policy=PUBLISH_RETRY,
            )
        except (asyncio.CancelledError, ActivityError) as err:
            if is_cancelled_exception(err):
                await self._finish_cancelled(task_id, manifest)
                raise
            await self._record_failure(task_id, err)
            # Re-raise the original error so it is the workflow's failure cause
            raise

        # 3. Artifacts are in MinIO and git; the local scratch copy is no longer needed
        try:
            await workflow.execute_activity(
                cleanup_scratch, task_id, start_to_close_timeout=SHORT_TIMEOUT, retry_policy=STANDARD_RETRY,
            )
        except ActivityError as err:
            workflow.logger.warning(f"Scratch cleanup failed for {task_id}: {_error_message(err)}")

        self._summary = {
            "task_id": task_id,
            "model_name": model_name,
            "status": "completed",
            "attempt": manifest["attempt"],
            "file_count": manifest["file_count"],
            "files": [f["rel_path"] for f in manifest["files"]],
            "publish": publish,
        }
        return self._summary

    async def _finish_cancelled(self, task_id: str, manifest: Optional[Dict[str, Any]]) -> None:
        """Publish completed runs saved before the cancel (if any), then record the outcome.
        Runs after the cancel was delivered, so these activities are not themselves cancelled."""
        self._summary["status"] = "cancelling"
        publish = None
        try:
            if manifest is None:
                # The eval activity was cancelled; it wrote its manifest to MinIO before returning
                latest = await workflow.execute_activity(
                    load_latest_manifest, task_id, start_to_close_timeout=SHORT_TIMEOUT, retry_policy=STANDARD_RETRY,
                )
                if latest and latest.get("status") == "cancelled":
                    manifest = latest
            if manifest and manifest.get("files"):
                publish = await workflow.execute_activity(
                    publish_results, manifest, start_to_close_timeout=SHORT_TIMEOUT, retry_policy=PUBLISH_RETRY,
                )
        except ActivityError as err:
            workflow.logger.warning(f"Publishing partial results for {task_id} failed: {_error_message(err)}")

        self._summary.update(
            status="cancelled",
            # Partial unless the eval had already completed when the cancel arrived
            partial=not manifest or manifest.get("status") != "completed",
            attempt=manifest.get("attempt") if manifest else None,
            file_count=len(manifest["files"]) if manifest else 0,
            files=[f["rel_path"] for f in manifest["files"]] if manifest else [],
            publish=publish,
        )
        try:
            await workflow.execute_activity(
                record_outcome, args=[task_id, self._summary], start_to_close_timeout=SHORT_TIMEOUT, retry_policy=STANDARD_RETRY,
            )
        except ActivityError as err:
            workflow.logger.warning(f"Recording cancellation for {task_id} failed: {_error_message(err)}")

    async def _record_failure(self, task_id: str, err: BaseException) -> None:
        """Best effort: write <task_id>/failure.json. Never replaces the original error."""
        cause = getattr(err, "cause", None) or err
        self._summary.update(status="failed", error=_error_message(err))
        record = {
            "status": "failed",
            "error": _error_message(err),
            "error_type": getattr(cause, "type", None) or type(cause).__name__,
            "retry_state": getattr(getattr(err, "retry_state", None), "name", None),
            "max_attempts": STANDARD_RETRY.maximum_attempts,
            "failed_at": workflow.now().isoformat(),
        }
        try:
            latest = await workflow.execute_activity(
                load_latest_manifest, task_id, start_to_close_timeout=SHORT_TIMEOUT, retry_policy=STANDARD_RETRY,
            )
            record["attempts"] = latest["attempt"] if latest else None
            await workflow.execute_activity(
                record_outcome, args=[task_id, record], start_to_close_timeout=SHORT_TIMEOUT, retry_policy=STANDARD_RETRY,
            )
        except ActivityError as record_err:
            workflow.logger.warning(f"Recording failure for {task_id} failed: {_error_message(record_err)}")
