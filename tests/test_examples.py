"""Execute shipped examples and quickstarts, including empty-result presentation."""
from pathlib import Path
import re
import runpy
import textwrap

import numpy as np
import pytest
import hjcdik

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_examples_run_from_another_directory(example, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(example), run_name="__main__")
    result = namespace["out"]
    assert 0 <= result["count"] <= 4
    assert result["joint_config"].shape == (result["count"], hjcdik.num_joints())
    assert np.isfinite(result["pos_errors"]).all()
    output = capsys.readouterr().out
    assert "(m)" not in output
    if result["count"]:
        assert "(mm)" in output and "(rad)" in output


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_examples_handle_no_candidates(example, monkeypatch, capsys):
    empty = {"count": 0, "joint_config": np.empty((0, hjcdik.num_joints())),
             "pose": np.empty((0, 7)), "pos_errors": np.empty(0), "ori_errors": np.empty(0)}
    monkeypatch.setattr(hjcdik, "generate_solutions", lambda *args, **kwargs: empty)
    runpy.run_path(str(example), run_name="__main__")
    assert "no " in capsys.readouterr().out.lower()


@pytest.mark.parametrize("path,section", [
    ("README.md", "## Quick Start"),
    ("docs/source/user_guide/getting_started/installation.md", "## Quickstart"),
    ("docs/source/user_guide/upgrading.md", "For example,"),
    ("docs/source/index.rst", "Quick start"),
    ("docs/source/api_reference/python.rst", ".. code-block:: python"),
])
def test_documented_quickstarts(path, section, tmp_path, monkeypatch):
    text = (ROOT / path).read_text().split(section, 1)[1]
    if path.endswith(".md"):
        code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    else:
        if ".. code-block:: python" in text:
            text = text.split(".. code-block:: python", 1)[1]
        code = re.match(r"\s*\n((?:   [^\n]*\n|\n)+)", text).group(1)
        code = textwrap.dedent(code)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(code, path, "exec"), namespace)
    result = namespace.get("out", namespace.get("result"))
    assert result["count"] > 0
    assert np.isfinite(result["pose"]).all()
