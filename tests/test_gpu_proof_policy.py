"""The local proof must cover implementation, dependencies, and the full collected suite."""
import subprocess
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")
proof = pytest.importorskip("pytest_gpu_proof.fingerprint")
yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]


def test_receipt_policy_cannot_narrow_the_source_scope():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["gpu_proof"]
    policy = yaml.safe_load((ROOT / "tests/gpu-proof-policy.yaml").read_text())
    assert policy["min_schema"] >= 3
    assert policy["allow_dirty"] is False
    assert sorted(policy["required_fingerprint_paths"]) == sorted(project["fingerprint_paths"])
    assert policy["required_fingerprint_extra_paths"] == []
    assert policy["required_fingerprint_excluded_paths"] == ["gpu-proof.json"]
    assert policy["required_test_manifest"] == "tests/gpu-proof-tests.txt"


def test_kernel_and_dependency_pin_changes_invalidate_fingerprint(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "csrc").mkdir()
    kernel = tmp_path / "csrc/kernel.cu"
    kernel.write_text("original kernel\n")
    git("add", "csrc/kernel.cu")
    git("update-index", "--add", "--cacheinfo", "160000," + "1" * 40 + ",external/GRiD")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["gpu_proof"]

    def fingerprint():
        return proof.compute_fingerprint(project["fingerprint_paths"], str(tmp_path))["digest"]

    baseline = fingerprint()
    kernel.write_text("changed kernel\n")
    assert fingerprint() != baseline
    kernel.write_text("original kernel\n")
    assert fingerprint() == baseline
    git("update-index", "--cacheinfo", "160000," + "2" * 40 + ",external/GRiD")
    assert fingerprint() != baseline
