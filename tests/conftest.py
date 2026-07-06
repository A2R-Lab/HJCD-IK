"""HJCD-IK test-suite conftest.

Wires pytest-gpu-proof into the suite. Every HJCD-IK test drives the CUDA kernel
(each module does ``pytest.importorskip("hjcdik")``), so *all* collected items are
GPU tests: we auto-tag every one ``gpu_proof`` so its outcome lands in the signed
receipt (``[tool.gpu_proof] required_marker = "gpu_proof"``). See
scripts/setup/run_gpu_proof.sh.

The vendored ``tests/pytest-gpu-proof`` submodule ships its OWN test suite (the
plugin's internals); we must not collect it as part of HJCD-IK's suite.
"""

import pytest

# Don't descend into the plugin submodule's own tests when collecting under tests/.
collect_ignore_glob = ["pytest-gpu-proof/*"]


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``gpu_proof`` marker to every HJCD-IK test.

    All tests require a built CUDA ``hjcdik`` (they ``importorskip`` it), so the
    receipt should cover the whole suite; add a test and it is covered
    automatically, with no per-test ``@pytest.mark.gpu_proof`` to remember.
    """
    for item in items:
        item.add_marker(pytest.mark.gpu_proof)
