"""HJCD-IK test-suite conftest.

Wires pytest-gpu-proof (installed from PyPI; see the ``[dev]`` extra) into the suite.
Every HJCD-IK test drives the CUDA kernel (each module does
``pytest.importorskip("hjcdik")``), so *all* collected items are GPU tests: we
auto-tag every one ``gpu_proof`` so its outcome lands in the signed receipt
(``[tool.gpu_proof] required_marker = "gpu_proof"``). See scripts/setup/run_gpu_proof.sh.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``gpu_proof`` marker to every HJCD-IK test.

    All tests require a built CUDA ``hjcdik`` (they ``importorskip`` it), so the
    receipt should cover the whole suite; add a test and it is covered
    automatically, with no per-test ``@pytest.mark.gpu_proof`` to remember.
    """
    for item in items:
        item.add_marker(pytest.mark.gpu_proof)
