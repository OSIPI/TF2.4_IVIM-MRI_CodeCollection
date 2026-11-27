import pathlib
import sys

import pytest

from src.wrappers.OsipiBase import OsipiBase

root = pathlib.Path(__file__).resolve().parents[3]


def set_sys_path(monkeypatch, entry=None):
    """Patch sys.path to a copy without the project root, with entry (if given) in front."""
    path = [p for p in sys.path if pathlib.Path(p).resolve() != root]
    if entry is not None:
        path.insert(0, entry)
    monkeypatch.setattr(sys, "path", path)


def test_root_as_empty_entry(monkeypatch):
    """'' in sys.path (python -c or a REPL started in the project root) is the current folder."""
    monkeypatch.chdir(root)
    set_sys_path(monkeypatch, "")
    assert type(OsipiBase(algorithm="IAR_LU_biexp")).__name__ == "IAR_LU_biexp"


def test_root_as_relative_entry(monkeypatch):
    """The tutorial notebook runs in doc/ and adds '..' to sys.path."""
    monkeypatch.chdir(root / "doc")
    set_sys_path(monkeypatch, "..")
    assert type(OsipiBase(algorithm="IAR_LU_biexp")).__name__ == "IAR_LU_biexp"


@pytest.mark.skipif(sys.platform != "win32", reason="paths are case-insensitive on Windows only")
def test_root_in_other_case(monkeypatch):
    """On Windows the root can be spelled in another case, e.g. with a lower-case drive letter."""
    set_sys_path(monkeypatch, str(root).swapcase())
    assert type(OsipiBase(algorithm="IAR_LU_biexp")).__name__ == "IAR_LU_biexp"


def test_root_missing_raises(monkeypatch, tmp_path):
    """Without the project root in sys.path the RuntimeError is still raised."""
    monkeypatch.chdir(tmp_path)
    set_sys_path(monkeypatch)
    with pytest.raises(RuntimeError, match="Root folder not found"):
        OsipiBase(algorithm="IAR_LU_biexp")
