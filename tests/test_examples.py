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


DOF = hjcdik.build_info()["num_joints"]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_examples_run_from_another_directory(example, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(example), run_name="__main__")
    result = namespace["out"][0]
    assert 0 <= result.count <= 4
    assert result.q.shape == (result.count, DOF)
    assert np.isfinite(result.pos_err_mm).all()
    output = capsys.readouterr().out
    assert "(m)" not in output
    if result.count:
        assert "(mm)" in output and "(rad)" in output


def _empty_result(S=4):
    nan = lambda *shape: np.full(shape, np.nan)
    return hjcdik.Result(q=nan(1, S, DOF), pose=nan(1, S, 7), pos_err_mm=nan(1, S), ori_err_rad=nan(1, S),
                         count=np.zeros(1, dtype=np.int64), valid=np.zeros((1, S), dtype=bool),
                         elapsed_ms=np.zeros(1), stats=None)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.stem)
def test_examples_handle_no_candidates(example, monkeypatch, capsys):
    monkeypatch.setattr(hjcdik.Solver, "solve", lambda self, *args, **kwargs: _empty_result())
    runpy.run_path(str(example), run_name="__main__")
    assert "no " in capsys.readouterr().out.lower()


@pytest.mark.parametrize("path,section", [
    ("README.md", "## Quick Start"),
    ("docs/source/user_guide/getting_started/installation.md", "## Quickstart"),
    ("docs/source/user_guide/upgrading.md", "## Migrating a call"),
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
    assert result.count.sum() > 0
    assert np.isfinite(result.pose[result.valid]).all()
