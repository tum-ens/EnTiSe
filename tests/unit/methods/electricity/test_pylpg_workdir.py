"""pyLPG runs in its own folder, never in the caller's working directory."""

import os
import sys
import types

import pandas as pd

from entise.methods.electricity.utils import run_year_chunk


def _fake_pylpg(monkeypatch, package_dir, calls):
    """Install a stand-in pylpg whose engine writes C1/ into the current directory, as pyLPG does."""

    def execute_lpg_tsib(**kwargs):
        cwd = os.getcwd()
        calls.append(cwd)
        os.makedirs(os.path.join(cwd, "C1"), exist_ok=True)
        idx = pd.date_range(kwargs["startdate"], kwargs["enddate"], freq="min", inclusive="left")
        return pd.DataFrame({"Electricity_HH1": 1.0}, index=idx)

    pylpg = types.ModuleType("pylpg")
    pylpg.__file__ = str(package_dir / "__init__.py")
    execution = types.ModuleType("pylpg.lpg_execution")
    execution.execute_lpg_tsib = execute_lpg_tsib
    bindings = types.ModuleType("pylpg.lpgpythonbindings")
    bindings.EnergyIntensityType = types.SimpleNamespace(Random="Random")
    pylpg.lpg_execution = execution
    pylpg.lpgpythonbindings = bindings
    monkeypatch.setitem(sys.modules, "pylpg", pylpg)
    monkeypatch.setitem(sys.modules, "pylpg.lpg_execution", execution)
    monkeypatch.setitem(sys.modules, "pylpg.lpgpythonbindings", bindings)


def test_run_year_chunk_keeps_pylpg_out_of_the_working_directory(tmp_path, monkeypatch):
    package_dir = tmp_path / "site-packages" / "pylpg"
    package_dir.mkdir(parents=True)  # no C1 template, as in current pyLPG releases
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr("tempfile.tempdir", None)
    calls = []
    _fake_pylpg(monkeypatch, package_dir, calls)

    df = run_year_chunk(
        obj_id="street/house",  # an id with a slash must not break the run folder
        year=2022,
        start=pd.Timestamp("2022-01-01 00:00"),
        end=pd.Timestamp("2022-01-01 01:00"),
        households=1,
        occupants_per_household=2,
        energy_intensity_name=None,
    )

    assert len(df) == 60
    assert os.getcwd() == str(caller)
    assert list(caller.iterdir()) == []
    assert calls and calls[0] != str(caller)
    # The run folder is removed afterwards.
    assert list((tmp_path / "tmp").iterdir()) == []
