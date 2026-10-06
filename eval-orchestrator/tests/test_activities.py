import dataclasses
import json
import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from temporalio.exceptions import ApplicationError, CancelledError
from temporalio.testing import ActivityEnvironment

import activities
from activities import (
    cleanup_expired_scratch,
    cleanup_scratch,
    git_blob_sha,
    load_latest_manifest,
    provision_model,
    publish_results,
    record_outcome,
    run_evaluation,
    verify_model_loaded,
)
from conftest import FakeEvaluationCancelled

PAYLOAD = {"model_name": "llama-3", "config": {"ctx_size": 4096}}


def write_run(output_dir, name="2026-07-31T04-39-55.json", content='{"score": 1}'):
    path = os.path.join(output_dir, "llama-3", "format-json-boolean", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def lemonade_ok(monkeypatch):
    """/load succeeds and /health reports the requested model as loaded."""
    loaded = {"model": None}

    def post(url, json, timeout):
        loaded["model"] = json["model_name"]
        return MagicMock()

    def get(url, timeout):
        return MagicMock(json=lambda: {"status": "ok", "model_loaded": loaded["model"], "all_models_loaded": []})

    monkeypatch.setattr(activities.requests, "post", post)
    monkeypatch.setattr(activities.requests, "get", get)
    return loaded


# --- provisioning helpers ---------------------------------------------------

def test_provision_model_posts_load():
    with patch("activities.requests.post") as mock_post:
        ActivityEnvironment().run(provision_model, "llama-3", {"ctx_size": 4096})
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["json"] == {"model_name": "llama-3", "ctx_size": 4096}
        assert mock_post.call_args.args[0].endswith("/load")


def test_provision_model_http_error():
    with patch("activities.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("Bad Request")
        with pytest.raises(requests.exceptions.HTTPError):
            ActivityEnvironment().run(provision_model, "llama-3")


@pytest.mark.parametrize("health, ok", [
    ({"model_loaded": "llama-3"}, True),
    ({"model_loaded": "judge", "all_models_loaded": [{"model_name": "judge"}, {"model_name": "llama-3"}]}, True),
    ({"model_loaded": "qwen", "all_models_loaded": [{"model_name": "qwen"}]}, False),
    ({"status": "ok"}, True),  # older server without loaded-model fields: skip the check
])
def test_verify_model_loaded(health, ok):
    with patch("activities.requests.get") as mock_get:
        mock_get.return_value.json.return_value = health
        if ok:
            ActivityEnvironment().run(verify_model_loaded, "llama-3")
        else:
            with pytest.raises(ApplicationError, match="not llama-3"):
                ActivityEnvironment().run(verify_model_loaded, "llama-3")


# --- run_evaluation -----------------------------------------------------------

def test_run_evaluation_success(evaluer_bench, scratch, fake_minio):
    def suite(model, output_dir, on_tick, should_cancel, poll_interval, grace_period):
        on_tick({"completed": 1, "total": 1})
        write_run(output_dir)
    evaluer_bench.run_evaluation_suite.side_effect = suite

    env = ActivityEnvironment()
    heartbeats = []
    env.on_heartbeat = lambda *d: heartbeats.append(d[0])
    manifest = env.run(run_evaluation, "test-task", PAYLOAD)

    kwargs = evaluer_bench.run_evaluation_suite.call_args.kwargs
    assert kwargs["model"] == "llama-3"
    assert kwargs["output_dir"] == os.path.join(str(scratch), "test-task", "attempt-1")
    assert manifest["status"] == "completed"
    assert manifest["files"] == [{
        "key": "test-task/attempt-1/llama-3/format-json-boolean/2026-07-31T04-39-55.json",
        "rel_path": "llama-3/format-json-boolean/2026-07-31T04-39-55.json",
    }]
    assert manifest["file_count"] == 1
    assert json.loads(fake_minio.objects["test-task/manifests/attempt-1.json"]) == manifest
    assert "eval-artifacts" in fake_minio.buckets
    # Heartbeats report progress from evaluerBench's on_tick
    assert {"phase": "evaluating", "attempt": 1, "completed": 1, "total": 1} in heartbeats


def test_run_evaluation_failure_uploads_but_reraises(evaluer_bench, scratch, fake_minio):
    def suite(model, output_dir, **_):
        write_run(output_dir)
        raise subprocess.CalledProcessError(1, "npm")
    evaluer_bench.run_evaluation_suite.side_effect = suite

    with pytest.raises(subprocess.CalledProcessError):
        ActivityEnvironment().run(run_evaluation, "test-task", PAYLOAD)
    # Kept in MinIO for debugging; the manifest says it failed, so it is never published
    assert "test-task/attempt-1/llama-3/format-json-boolean/2026-07-31T04-39-55.json" in fake_minio.objects
    assert json.loads(fake_minio.objects["test-task/manifests/attempt-1.json"])["status"] == "failed"


def test_run_evaluation_upload_error_does_not_hide_eval_error(evaluer_bench, scratch, monkeypatch):
    evaluer_bench.run_evaluation_suite.side_effect = subprocess.CalledProcessError(2, "npm")
    monkeypatch.setattr(activities, "get_minio_client", MagicMock(side_effect=ConnectionError("minio down")))
    with pytest.raises(subprocess.CalledProcessError):
        ActivityEnvironment().run(run_evaluation, "test-task", PAYLOAD)


def test_run_evaluation_isolates_attempts_and_logs_previous_progress(evaluer_bench, scratch, fake_minio):
    evaluer_bench.run_evaluation_suite.side_effect = lambda model, output_dir, **_: write_run(output_dir, "b.json")
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, attempt=2, heartbeat_details=[{"phase": "evaluating", "completed": 3}])
    stale = scratch / "test-task" / "attempt-1"
    stale.mkdir(parents=True)
    (stale / "a.json").write_text("{}")

    with patch.object(activities.activity, "logger") as logger:
        manifest = env.run(run_evaluation, "test-task", PAYLOAD)
    assert [f["key"] for f in manifest["files"]] == ["test-task/attempt-2/llama-3/format-json-boolean/b.json"]
    assert any("'completed': 3" in str(c) for c in logger.info.call_args_list)


def test_run_evaluation_cancelled_uploads_completed_runs(evaluer_bench, scratch, fake_minio):
    env = ActivityEnvironment()
    seen = {}

    def suite(model, output_dir, on_tick, should_cancel, **_):
        write_run(output_dir, "done.json", '{"cancelled": true}')
        on_tick({"completed": 1, "total": 3})
        env.cancel()  # what the worker does when the heartbeat response says "cancel requested"
        deadline = time.monotonic() + 5
        while not should_cancel():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        seen["should_cancel"] = True
        raise FakeEvaluationCancelled()
    evaluer_bench.run_evaluation_suite.side_effect = suite

    with pytest.raises(CancelledError):
        env.run(run_evaluation, "test-task", PAYLOAD)
    assert seen["should_cancel"]
    manifest = json.loads(fake_minio.objects["test-task/manifests/attempt-1.json"])
    assert manifest["status"] == "cancelled"
    assert [f["rel_path"] for f in manifest["files"]] == ["llama-3/format-json-boolean/done.json"]


def test_run_evaluation_verify_mismatch_fails_attempt(evaluer_bench, scratch, fake_minio, monkeypatch):
    monkeypatch.setattr(activities, "LEMONADE_VERIFY_LOADED", True)
    monkeypatch.setattr(activities.requests, "get", lambda url, timeout: MagicMock(json=lambda: {"model_loaded": "other"}))
    with pytest.raises(ApplicationError, match="not llama-3"):
        ActivityEnvironment().run(run_evaluation, "test-task", PAYLOAD)
    evaluer_bench.run_evaluation_suite.assert_not_called()


def test_run_evaluation_rejects_path_like_task_id(evaluer_bench, scratch, fake_minio):
    with pytest.raises(ApplicationError, match="Invalid task_id"):
        ActivityEnvironment().run(run_evaluation, "../etc", PAYLOAD)


# --- publish_results ----------------------------------------------------------

def manifest_for(fake_minio, files, attempt=2, status="completed"):
    entries = []
    for rel_path, data in files.items():
        key = f"test-task/attempt-{attempt}/{rel_path}"
        fake_minio.objects[key] = data
        entries.append({"key": key, "rel_path": rel_path})
    return {"task_id": "test-task", "attempt": attempt, "status": status, "files": entries, "file_count": len(entries)}


def test_git_blob_sha_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_publish_results_publishes_only_manifest_files(fake_minio, fake_repo):
    fake_minio.objects["test-task/attempt-1/llama-3/format-json-boolean/stale.json"] = b"from failed attempt"
    manifest = manifest_for(fake_minio, {
        "llama-3/format-json-boolean/2026-07-31T04-39-55.json": b'{"hello": "world"}',
        "llama-3/visual-tictactoe-game/artifact-2026-07-31T04-41-02.png": b"\x89PNG\xff\xfe",
    })

    result = ActivityEnvironment().run(publish_results, manifest)

    assert result["status"] == "published"
    assert len(fake_repo.ref_updates) == 1
    files = fake_repo.head_files()
    assert files == {
        "src/resources/evaluerBench/llama-3/format-json-boolean/2026-07-31T04-39-55.json": b'{"hello": "world"}',
        "src/resources/evaluerBench/llama-3/visual-tictactoe-game/artifact-2026-07-31T04-41-02.png": b"\x89PNG\xff\xfe",
    }


def test_publish_results_is_idempotent(fake_minio, fake_repo):
    manifest = manifest_for(fake_minio, {"llama-3/format-json-boolean/run.json": b"{}"})
    env = ActivityEnvironment()

    first = env.run(publish_results, manifest)
    second = env.run(publish_results, manifest)

    assert first["status"] == "published"
    assert second == {"status": "already_published", "commit": first["commit"], "paths": first["paths"]}
    assert len(fake_repo.ref_updates) == 1


def test_publish_results_conflict_is_retryable_and_rebuilds_on_new_head(fake_minio, fake_repo):
    manifest = manifest_for(fake_minio, {"llama-3/format-json-boolean/run.json": b"{}"})
    fake_repo.conflicts = 1
    env = ActivityEnvironment()

    with pytest.raises(ApplicationError) as exc_info:
        env.run(publish_results, manifest)
    assert exc_info.value.type == "PublishConflict"
    assert not exc_info.value.non_retryable
    concurrent_head = fake_repo.head

    result = env.run(publish_results, manifest)  # what Temporal's retry does

    assert result["status"] == "published"
    assert fake_repo.commits[result["commit"]]["parents"] == [concurrent_head]
    assert set(fake_repo.head_files()) == {"src/other.txt", "src/resources/evaluerBench/llama-3/format-json-boolean/run.json"}


def test_publish_results_marks_cancelled_commits(fake_minio, fake_repo):
    manifest = manifest_for(fake_minio, {"llama-3/format-json-boolean/run.json": b"{}"}, status="cancelled")
    result = ActivityEnvironment().run(publish_results, manifest)
    assert "[partial: cancelled]" in fake_repo.commits[result["commit"]]["message"]


def test_publish_results_missing_env():
    with patch("activities.GITHUB_TOKEN", None), patch("activities.GITHUB_REPO", None):
        with pytest.raises(ValueError, match="Missing GITHUB_TOKEN or GITHUB_REPO environment variables"):
            ActivityEnvironment().run(publish_results, {"task_id": "test-task", "files": []})


def test_publish_results_empty_manifest_is_non_retryable(fake_minio, fake_repo):
    with pytest.raises(ApplicationError, match="no files to publish") as exc_info:
        ActivityEnvironment().run(publish_results, {"task_id": "test-task", "files": []})
    assert exc_info.value.non_retryable


# --- outcome records and scratch cleanup --------------------------------------

def test_load_latest_manifest_picks_highest_attempt(fake_minio):
    for n in (1, 2, 10):
        fake_minio.objects[f"test-task/manifests/attempt-{n}.json"] = json.dumps({"attempt": n}).encode()
    assert ActivityEnvironment().run(load_latest_manifest, "test-task") == {"attempt": 10}
    assert ActivityEnvironment().run(load_latest_manifest, "other-task") is None


def test_record_outcome_writes_failure_json(fake_minio):
    key = ActivityEnvironment().run(record_outcome, "test-task", {"status": "failed", "error": "boom", "attempts": 3})
    assert key == "test-task/failure.json"
    record = json.loads(fake_minio.objects[key])
    assert record["error"] == "boom" and record["attempts"] == 3 and "recorded_at" in record


def test_cleanup_scratch_removes_task_dir(scratch):
    (scratch / "test-task" / "attempt-1").mkdir(parents=True)
    assert ActivityEnvironment().run(cleanup_scratch, "test-task") is True
    assert not (scratch / "test-task").exists()
    assert ActivityEnvironment().run(cleanup_scratch, "test-task") is False


def test_cleanup_expired_scratch(tmp_path):
    old = time.time() - 10 * 86400
    for name, attempt_dir in [("old-task", True), ("keep-me", True), ("not-ours", False), ("new-task", True)]:
        path = tmp_path / name
        (path / ("attempt-1" if attempt_dir else "data")).mkdir(parents=True)
        if name != "new-task":
            os.utime(path, (old, old))

    removed = ActivityEnvironment().run(cleanup_expired_scratch, str(tmp_path), 7, "keep-me")

    assert removed == ["old-task"]
    assert sorted(os.listdir(tmp_path)) == ["keep-me", "new-task", "not-ours"]
    assert ActivityEnvironment().run(cleanup_expired_scratch, str(tmp_path), 0) == []
