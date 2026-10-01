"""Exercise generated-init failure propagation through the installed Python extension.

Linux LD_PRELOAD tests use shared cudart (our build contract); no real OOM, reset,
kernel poisoning, or timing. GRiD owns broader generated-helper fault coverage.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cuda_fault_shim(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("LD_PRELOAD consumer fault injection requires Linux")
    compiler = shutil.which("c++")
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cuda = Path(nvcc).resolve().parent.parent
    assert compiler and (cuda / "include/cuda_runtime_api.h").is_file(), (
        "consumer fault tests require a C++ compiler and CUDA development headers"
    )
    library = tmp_path_factory.mktemp("cuda-fault") / "fault.so"
    run = subprocess.run([
        compiler, "-std=c++17", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        str(ROOT / "tests/native/cuda_fault_shim.cpp"), "-o", str(library),
        "-I" + str(cuda / "include"), "-L" + str(cuda / "lib64"),
        "-Wl,-rpath," + str(cuda / "lib64"), "-Wl,--no-as-needed", "-lcudart", "-ldl",
    ], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    return library


@pytest.mark.parametrize("entrypoint", ["sample", "solve"])
@pytest.mark.parametrize("nth,operation", [
    (1, "cudaMalloc(d_joint_limits)"),
    (2, "cudaMemcpy(d_joint_limits)"),
    (3, "cudaMalloc(d_XImats)"),
    (4, "cudaMemcpy(d_XImats)"),
    (5, "cudaMalloc(d_robotModel)"),
    (6, "cudaMemcpy(d_robotModel)"),
])
def test_checked_initialization_raises_cleans_up_and_retries(cuda_fault_shim, nth, operation, entrypoint):
    # A fresh process per case keeps caches empty and makes an accidental exit observable.
    code = r"""
import ctypes
import sys
import hjcdik
shim = ctypes.CDLL(sys.argv[1])
nth, operation, entrypoint = int(sys.argv[2]), sys.argv[3], sys.argv[4]
shim.hjcd_test_arm.argtypes = [ctypes.c_int]
shim.hjcd_test_arm.restype = None
assert hjcdik.num_joints() == 7, "fault ordering assumes the default Panda model"
def call():
    if entrypoint == "sample":
        return hjcdik.sample_targets(1, seed=17)
    return hjcdik.generate_solutions([0.3, 0, 0.5, 1, 0, 0, 0],
                                     batch_size=32, refine_fp64=0)
assert shim.hjcd_test_outstanding() == 0
for attempt in range(3):
    # After a model failure the limits were successfully initialized once, so those
    # first two runtime calls are correctly omitted on the following attempts.
    shim.hjcd_test_arm(nth - 2 if attempt and nth > 2 else nth)
    try:
        call()
    except RuntimeError as error:
        assert "GRiD CUDA initialization failed" in str(error), str(error)
        assert operation in str(error), str(error)
    else:
        raise AssertionError("injected initialization error did not reach Python")
    assert shim.hjcd_test_injected() == 1, "interposition/fault ordering did not work"
    assert shim.hjcd_test_outstanding() == 0, "partial initialization leaked"
    assert shim.hjcd_test_resets() == 0
shim.hjcd_test_arm(0)
result = call()
assert len(result) == 1 if entrypoint == "sample" else result["count"] == 1
retained = shim.hjcd_test_outstanding()
assert retained == 2, "only the successful model and its nested transforms remain cached"
call()
assert shim.hjcd_test_outstanding() == retained, "repeat call leaked/recreated a cached model"
assert shim.hjcd_test_resets() == 0
print("checked initialization recovered")
"""
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(cuda_fault_shim)
    run = subprocess.run(
        [sys.executable, "-c", code, str(cuda_fault_shim), str(nth), operation, entrypoint],
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "checked initialization recovered" in run.stdout
