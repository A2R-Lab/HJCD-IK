"""HJCD-IK test-suite conftest.

Wires pytest-gpu-proof (installed from PyPI; see the ``[dev]`` extra) into the suite.
The suite includes CUDA correctness and host/codegen contract tests. We
auto-tag every collected item ``gpu_proof`` so the full suite lands in the signed receipt
(``[tool.gpu_proof] required_marker = "gpu_proof"``). See scripts/setup/run_gpu_proof.sh.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``gpu_proof`` marker to every HJCD-IK test.

    The receipt covers the entire suite, including host/codegen contracts;
    add a test and it is covered automatically.
    """
    for item in items:
        item.add_marker(pytest.mark.gpu_proof)
