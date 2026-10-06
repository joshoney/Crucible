import asyncio
import concurrent.futures
import json
import os
import subprocess
import threading
import time
import uuid
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from temporalio import activity
from temporalio.client import WorkflowExecutionStatus, WorkflowFailureError
from temporalio.exceptions import ActivityError, ApplicationError, CancelledError, TimeoutError, TimeoutType
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

import activities
from conftest import FakeEvaluationCancelled
from workflow import LlmEvaluationWorkflow

TASK_QUEUE = "eval-task-queue"
RUN_FILE = "llama-3/format-json-boolean/2026-07-31T04-39-55.json"

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


def worker(env, acts, **kwargs):
    return Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[LlmEvaluationWorkflow],
        activities=acts,
        activity_executor=concurrent.futures.ThreadPoolExecutor(max_workers=4),
        max_concurrent_activities=1,
        # Heartbeats carry cancellation to sync activities; don't wait the default 60s for one
        max_heartbeat_throttle_interval=timedelta(milliseconds=200),
        **kwargs,
    )


class Recorder:
    """Mock versions of the side-effect activities, recording their inputs."""

    def __init__(self):
        self.published = []
        self.outcomes = []
        self.cleaned = []

        @activity.defn(name="publish_results")
        def publish_results(manifest: dict) -> dict:
            self.published.append(manifest)
            return {"status": "published", "commit": "abc123", "paths": [f["rel_path"] for f in manifest["files"]]}

        @activity.defn(name="record_outcome")
        def record_outcome(task_id: str, record: dict) -> str:
            self.outcomes.append(record)
            return f"{task_id}/{record['status']}.json"

        @activity.defn(name="cleanup_scratch")
        def cleanup_scratch(task_id: str) -> bool:
            self.cleaned.append(task_id)
            return True

        @activity.defn(name="load_latest_manifest")
        def load_latest_manifest(task_id: str):
            return None

        self.publish_results = publish_results
        self.record_outcome = record_outcome
        self.cleanup_scratch = cleanup_scratch
        self.load_latest_manifest = load_latest_manifest

    def side_effects(self):
        return [self.publish_results, self.record_outcome, self.cleanup_scratch, self.load_latest_manifest]


def fake_manifest(task_id, attempt=1):
    return {
        "task_id": task_id, "model_name": "llama-3", "attempt": attempt, "status": "completed",
        "files": [{"key": f"{task_id}/attempt-{attempt}/{RUN_FILE}", "rel_path": RUN_FILE}], "file_count": 1,
    }


def write_run(output_dir, rel_path=RUN_FILE, content="{}"):
    path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture
def real_eval_env(evaluer_bench, scratch, fake_minio, monkeypatch):
    """The real run_evaluation activity against fake Lemonade, MinIO and evaluerBench."""
    lemonade = {"loaded": None}

    def post(url, json, timeout):
        lemonade["loaded"] = json["model_name"]
        return MagicMock()

    def get(url, timeout):
        return MagicMock(json=lambda: {"model_loaded": lemonade["loaded"], "all_models_loaded": [{"model_name": lemonade["loaded"]}]})

    monkeypatch.setattr(activities.requests, "post", post)
    monkeypatch.setattr(activities.requests, "get", get)
    return lemonade


async def test_success_publishes_once_and_cleans_up(env):
    rec = Recorder()

    @activity.defn(name="run_evaluation")
    async def run_evaluation(task_id: str, payload: dict) -> dict:
        return fake_manifest(task_id)

    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    async with worker(env, [run_evaluation, *rec.side_effects()]):
        result = await env.client.execute_workflow(
            LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=task_id, task_queue=TASK_QUEUE,
        )
    assert result["status"] == "completed"
    assert result["files"] == [RUN_FILE]
    assert result["publish"]["commit"] == "abc123"
    assert rec.published == [fake_manifest(task_id)]
    assert rec.cleaned == [task_id]
    assert rec.outcomes == []


async def test_eval_failure_is_the_workflow_cause_and_nothing_is_published(env):
    rec = Recorder()
    calls = []

    @activity.defn(name="run_evaluation")
    async def run_evaluation(task_id: str, payload: dict) -> dict:
        calls.append(activity.info().attempt)
        raise ApplicationError("Simulated failure")

    async with worker(env, [run_evaluation, *rec.side_effects()]):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=f"eval-test-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE,
            )
    assert isinstance(exc_info.value.cause, ActivityError)
    assert "Simulated failure" in str(exc_info.value.cause.cause)
    assert calls == [1, 2, 3]
    assert rec.published == []
    assert rec.cleaned == []
    assert [o["status"] for o in rec.outcomes] == ["failed"]
    assert rec.outcomes[0]["error"] == "Simulated failure"


async def test_failure_recording_error_does_not_mask_eval_error(env):
    rec = Recorder()

    @activity.defn(name="run_evaluation")
    async def run_evaluation(task_id: str, payload: dict) -> dict:
        raise ApplicationError("Simulated failure", non_retryable=True)

    @activity.defn(name="record_outcome")
    async def record_outcome(task_id: str, record: dict) -> str:
        raise ApplicationError("MinIO down", non_retryable=True)

    async with worker(env, [run_evaluation, record_outcome, rec.publish_results, rec.cleanup_scratch, rec.load_latest_manifest]):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=f"eval-test-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE,
            )
    assert "Simulated failure" in str(exc_info.value.cause.cause)


async def test_only_the_successful_attempts_files_are_published(env, real_eval_env, evaluer_bench, fake_minio):
    rec = Recorder()

    def suite(model, output_dir, **_):
        if output_dir.endswith("attempt-1"):
            write_run(output_dir, "llama-3/format-json-boolean/partial.json")
            raise subprocess.CalledProcessError(1, "npm")
        write_run(output_dir)
    evaluer_bench.run_evaluation_suite.side_effect = suite

    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    async with worker(env, [activities.run_evaluation, *rec.side_effects()]):
        await env.client.execute_workflow(
            LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=task_id, task_queue=TASK_QUEUE,
        )

    assert f"{task_id}/attempt-1/llama-3/format-json-boolean/partial.json" in fake_minio.objects
    [manifest] = rec.published
    assert manifest["attempt"] == 2
    assert manifest["files"] == [{"key": f"{task_id}/attempt-2/{RUN_FILE}", "rel_path": RUN_FILE}]


async def test_eval_that_stops_heartbeating_times_out(env, monkeypatch):
    # The time-skipping server doesn't skip time while an activity runs, so shrink the
    # heartbeat timeout; the unsandboxed runner makes the patched module value visible
    import workflow
    assert workflow.EVAL_HEARTBEAT_TIMEOUT == timedelta(minutes=2)
    monkeypatch.setattr(workflow, "EVAL_HEARTBEAT_TIMEOUT", timedelta(seconds=1))
    rec = Recorder()

    @activity.defn(name="run_evaluation")
    async def run_evaluation(task_id: str, payload: dict) -> dict:
        activity.heartbeat({"phase": "provisioning"})
        await asyncio.sleep(3600)  # hung: no more heartbeats
        return fake_manifest(task_id)

    async with worker(env, [run_evaluation, *rec.side_effects()], workflow_runner=UnsandboxedWorkflowRunner()):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await env.client.execute_workflow(
                LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=f"eval-test-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE,
            )
    cause = exc_info.value.cause.cause
    assert isinstance(cause, TimeoutError)
    assert cause.type == TimeoutType.HEARTBEAT
    assert rec.published == []


async def test_cancel_mid_run_publishes_completed_runs_as_partial(env, real_eval_env, evaluer_bench, fake_minio):
    rec = Recorder()
    started = threading.Event()
    seen = {}

    def suite(model, output_dir, on_tick, should_cancel, **_):
        write_run(output_dir, content='{"cancelled": true}')  # first eval finished before the cancel
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            on_tick({"completed": 1, "total": 3})
            started.set()
            if should_cancel():
                seen["should_cancel"] = True
                raise FakeEvaluationCancelled()
            time.sleep(0.05)
        raise AssertionError("cancellation never reached evaluerBench")
    evaluer_bench.run_evaluation_suite.side_effect = suite

    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    acts = [activities.run_evaluation, activities.load_latest_manifest, activities.record_outcome, rec.publish_results, rec.cleanup_scratch]
    async with worker(env, acts):
        handle = await env.client.start_workflow(
            LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=task_id, task_queue=TASK_QUEUE,
        )
        assert await asyncio.get_running_loop().run_in_executor(None, started.wait, 10)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError) as exc_info:
            await handle.result()
        assert isinstance(exc_info.value.cause, CancelledError)
        assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED
        summary = await handle.query(LlmEvaluationWorkflow.summary)

    assert seen["should_cancel"]
    [manifest] = rec.published
    assert manifest["status"] == "cancelled"
    assert manifest["files"] == [{"key": f"{task_id}/attempt-1/{RUN_FILE}", "rel_path": RUN_FILE}]
    assert fake_minio.objects[f"{task_id}/attempt-1/{RUN_FILE}"] == b'{"cancelled": true}'
    assert json.loads(fake_minio.objects[f"{task_id}/cancelled.json"])["partial"] is True
    assert summary["status"] == "cancelled"
    assert rec.cleaned == []  # cancelled runs keep their scratch dir


async def test_cancel_with_no_completed_runs_publishes_nothing(env, real_eval_env, evaluer_bench, fake_minio):
    rec = Recorder()
    started = threading.Event()

    def suite(model, output_dir, on_tick, should_cancel, **_):
        while not should_cancel():
            on_tick({"completed": 0, "total": 3})
            started.set()
            time.sleep(0.05)
        raise FakeEvaluationCancelled()
    evaluer_bench.run_evaluation_suite.side_effect = suite

    task_id = f"eval-test-{uuid.uuid4().hex[:8]}"
    acts = [activities.run_evaluation, activities.load_latest_manifest, activities.record_outcome, rec.publish_results, rec.cleanup_scratch]
    async with worker(env, acts):
        handle = await env.client.start_workflow(
            LlmEvaluationWorkflow.run, {"model_name": "llama-3"}, id=task_id, task_queue=TASK_QUEUE,
        )
        assert await asyncio.get_running_loop().run_in_executor(None, started.wait, 10)
        await handle.cancel()
        with pytest.raises(WorkflowFailureError):
            await handle.result()
        assert (await handle.describe()).status == WorkflowExecutionStatus.CANCELED

    assert rec.published == []
    assert f"{task_id}/cancelled.json" in fake_minio.objects


async def test_concurrent_workflows_each_evaluate_their_own_model(env, real_eval_env, evaluer_bench):
    rec = Recorder()
    observed = []

    def suite(model, output_dir, on_tick, **_):
        # What a real eval would hit: whichever model Lemonade has loaded right now
        loaded_at_start = real_eval_env["loaded"]
        on_tick({"completed": 0, "total": 1})
        time.sleep(0.2)
        observed.append((model, loaded_at_start, real_eval_env["loaded"]))
        write_run(output_dir, f"{model}/format-json-boolean/run.json")
    evaluer_bench.run_evaluation_suite.side_effect = suite

    models = ["llama-3", "qwen-2.5", "phi-4"]
    async with worker(env, [activities.run_evaluation, *rec.side_effects()]):
        handles = [
            await env.client.start_workflow(
                LlmEvaluationWorkflow.run, {"model_name": m}, id=f"eval-{m}-{uuid.uuid4().hex[:8]}", task_queue=TASK_QUEUE,
            )
            for m in models
        ]
        # Real time: with queued activities and no timers, the test server would skip ahead
        with env.auto_time_skipping_disabled():
            results = await asyncio.gather(*(h.result() for h in handles))

    assert sorted(o[0] for o in observed) == sorted(models)
    for model, loaded_at_start, loaded_at_end in observed:
        assert model == loaded_at_start == loaded_at_end
    assert [r["model_name"] for r in results] == models

