"""The document cache is independently testable without CUDA."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_exact_document_cache_and_failure_recovery(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    binary = tmp_path / "problem_document"
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-UNDEBUG",
        "-I", str(ROOT / "csrc"), str(ROOT / "tests/native/test_problem_document.cpp"),
        "-o", str(binary),
    ], check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
