"""Runs the GPU notebook's code on CPU, against the fake cuDSS.

`notebooks/colab_gpu_tests.ipynb` can only run for real on a CUDA machine, so a mistake in
its code, like a stale name after a refactor, would otherwise only show up on the next
manual Colab run. These tests pull the code cells out of the notebook and run the sanity
check in-process, with the platform reported as a GPU and `spineax.cudss` replaced by the
fake from `fake_cudss.py`. That checks everything the cell asserts about `splineax`: which
solver `AutoSparseLinearSolver` picks, and that a solve through it is correct. It cannot
check the real CUDA stack, which is what the cell does on a GPU.
"""

import ast
import json
import runpy
from pathlib import Path

import jax
import pytest

import splineax.solvers._auto as _auto_module

from ..fake_cudss import FakeCuDSS

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "colab_gpu_tests.ipynb"
SANITY_CHECK_FILE = "gpu_sanity_check.py"


def _code_cells() -> list[str]:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def _written_script(filename: str) -> str:
    """The body of the cell that starts with `%%writefile <filename>`."""
    magic = f"%%writefile {filename}\n"
    (script,) = [cell for cell in _code_cells() if cell.startswith(magic)]
    return script.removeprefix(magic)


@pytest.mark.parametrize("cell", _code_cells())
def test_notebook_cells_are_valid_python(cell: str) -> None:
    """Every code cell parses, once Colab's `%%writefile` and `%` magics are set aside."""
    lines = [line for line in cell.splitlines() if not line.startswith("%")]
    ast.parse("\n".join(lines))


def test_sanity_check_passes_on_a_fake_gpu(
    fake_cudss: FakeCuDSS,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The sanity check cell picks `CuDSS` through `AutoSparseLinearSolver` and solves."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    script = tmp_path / SANITY_CHECK_FILE
    script.write_text(_written_script(SANITY_CHECK_FILE))

    runpy.run_path(str(script), run_name="__main__")

    output = capsys.readouterr().out
    assert "iterative refinement around CuDSS" in output
    assert "cuDSS is installed, selected, and solving on this GPU." in output
    assert fake_cudss.factorize_calls, "the solve never reached the fake cuDSS"
    assert fake_cudss.solve_calls, "the solve never reached the fake cuDSS"


def test_sanity_check_fails_without_cudss(
    fake_cudss: FakeCuDSS, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On a machine where `Auto` falls back to another solver, the cell's assert fires."""
    del fake_cudss
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: False)
    script = tmp_path / SANITY_CHECK_FILE
    script.write_text(_written_script(SANITY_CHECK_FILE))

    with pytest.raises(AssertionError, match="expected CuDSS"):
        runpy.run_path(str(script), run_name="__main__")
