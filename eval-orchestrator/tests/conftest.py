"""Shared fakes: evaluerBench (stubbed in sys.modules before activities is imported), MinIO, GitHub."""
import base64
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from github import GithubException, UnknownObjectException


class FakeEvaluationCancelled(Exception):
    """Real exception class standing in for evaluerBench.main.EvaluationCancelled."""


mock_evaluer_bench = MagicMock()
mock_evaluer_bench.EvaluationCancelled = FakeEvaluationCancelled
sys.modules["evaluerBench"] = mock_evaluer_bench
sys.modules["evaluerBench.main"] = mock_evaluer_bench

import activities


@pytest.fixture
def evaluer_bench():
    mock_evaluer_bench.run_evaluation_suite.reset_mock(side_effect=True, return_value=True)
    return mock_evaluer_bench


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    # The /app/scratchpad default isn't writable on CI runners
    root = tmp_path / "scratchpad"
    monkeypatch.setattr(activities, "SCRATCHPAD_ROOT", str(root))
    return root


class FakeMinio:
    def __init__(self):
        self.objects = {}
        self.buckets = set()

    def bucket_exists(self, bucket):
        return bucket in self.buckets

    def make_bucket(self, bucket):
        self.buckets.add(bucket)

    def fput_object(self, bucket, key, path):
        with open(path, "rb") as f:
            self.objects[key] = f.read()

    def put_object(self, bucket, key, data, length, content_type=None):
        self.objects[key] = data.read(length)

    def get_object(self, bucket, key):
        data = self.objects[key]
        return SimpleNamespace(read=lambda: data, close=lambda: None, release_conn=lambda: None)

    def list_objects(self, bucket, prefix="", recursive=False):
        return [SimpleNamespace(object_name=k) for k in sorted(self.objects) if k.startswith(prefix)]


@pytest.fixture
def fake_minio(monkeypatch):
    client = FakeMinio()
    monkeypatch.setattr(activities, "get_minio_client", lambda: client)
    return client


class FakeRepo:
    """Just enough of the GitHub Git Data API: commits are {path: bytes} snapshots."""

    def __init__(self, files=None):
        self.commits = {"c0": {"files": dict(files or {}), "parents": [], "message": "initial"}}
        self.trees = {"t-c0": dict(files or {})}
        self.blobs = {}
        self.head = "c0"
        self.ref_updates = []
        self.conflicts = 0  # next N ref updates lose a race to a concurrent commit
        self._n = 0

    def _id(self, prefix):
        self._n += 1
        return f"{prefix}{self._n}"

    def commit_directly(self, files, message="concurrent commit"):
        snapshot = {**self.commits[self.head]["files"], **files}
        sha = self._id("c")
        self.commits[sha] = {"files": snapshot, "parents": [self.head], "message": message}
        self.trees[f"t-{sha}"] = snapshot
        self.head = sha
        return sha

    def get_git_ref(self, name):
        repo = self
        head = self.head

        class Ref:
            object = SimpleNamespace(sha=head)

            def edit(self, sha, force=False):
                if repo.conflicts:
                    repo.conflicts -= 1
                    repo.commit_directly({"src/other.txt": b"someone else"})
                if not force and repo.commits[sha]["parents"] != [repo.head]:
                    raise GithubException(422, {"message": "Update is not a fast forward"}, None)
                repo.head = sha
                repo.ref_updates.append(sha)
        return Ref()

    def get_git_commit(self, sha):
        return SimpleNamespace(sha=sha, tree=SimpleNamespace(sha=f"t-{sha}"))

    def get_git_tree(self, sha):
        return SimpleNamespace(sha=sha)

    def get_contents(self, path, ref):
        files = self.commits[ref]["files"]
        if path not in files:
            raise UnknownObjectException(404, {"message": "Not Found"}, None)
        return SimpleNamespace(sha=activities.git_blob_sha(files[path]))

    def create_git_blob(self, content, encoding):
        data = base64.b64decode(content)
        sha = activities.git_blob_sha(data)
        self.blobs[sha] = data
        return SimpleNamespace(sha=sha)

    def create_git_tree(self, elements, base_tree):
        snapshot = dict(self.trees[base_tree.sha])
        for el in elements:
            ident = el._identity
            snapshot[ident["path"]] = ident["content"].encode("utf-8") if "content" in ident else self.blobs[ident["sha"]]
        sha = self._id("t-new")
        self.trees[sha] = snapshot
        return SimpleNamespace(sha=sha)

    def create_git_commit(self, message, tree, parents):
        sha = self._id("c")
        self.commits[sha] = {"files": self.trees[tree.sha], "parents": [p.sha for p in parents], "message": message}
        self.trees[f"t-{sha}"] = self.trees[tree.sha]
        return SimpleNamespace(sha=sha)

    def head_files(self):
        return self.commits[self.head]["files"]


@pytest.fixture
def fake_repo(monkeypatch):
    repo = FakeRepo()
    monkeypatch.setattr(activities, "Github", lambda auth: SimpleNamespace(get_repo=lambda name: repo))
    monkeypatch.setattr(activities, "GITHUB_TOKEN", "fake_token")
    monkeypatch.setattr(activities, "GITHUB_REPO", "fake/repo")
    return repo
