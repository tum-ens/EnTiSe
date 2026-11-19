"""Generating the same object again must give bit-identical time series.

The result of an object may not depend on the order of the objects, on the number of
workers, on the batch size, or on the process that happens to calculate it.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from entise import perf
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.generator import Generator

HERE = os.path.dirname(__file__)
HOURS = 14 * 24

# How the objects are run: (workers, batch_size, reverse the object order)
REFERENCE = (1, None, False)
VARIANTS = {
    "reversed order": (1, None, True),
    "two workers, one object per batch": (2, 1, False),
    "two workers, even batches, reversed order": (2, None, True),
}


def _weather() -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"), parse_dates=True)
    return weather.iloc[:HOURS].reset_index(drop=True)


def _hvac_1r1c():
    folder = os.path.join(HERE, "hvac_1r1c")
    objects = pd.read_csv(os.path.join(folder, "objects.csv"))
    objects = objects.loc[objects["hvac"] == "1R1C"]
    data = {"weather": _weather()}
    for name in ("windows", "internal_gains", "ventilation"):
        frame = pd.read_csv(os.path.join(folder, "data", f"{name}.csv"), parse_dates=True)
        data[name] = frame.iloc[:HOURS] if len(frame) > HOURS else frame
    return objects, data


def _hp_ruhnau():
    objects = pd.DataFrame(
        {
            O.ID: [1, 2, 3],
            "hp": "ruhnau",
            O.WEATHER: "weather",
            "hp_source": ["air", "soil", "water"],
            "hp_sink": ["radiator", "floor", "radiator"],
        }
    )
    return objects, {"weather": _weather()}


def _dhw_jordanvajen():
    objects = pd.read_csv(os.path.join(HERE, "dhw_jordanvajen", "objects.csv"))
    objects[O.SEED] = 7
    return objects, {"weather": _weather()}


def _pv_pvlib():
    objects = pd.read_csv(os.path.join(HERE, "pv_pvlib", "objects.csv"))
    return objects, {"weather": _weather()}


CASES = {
    "hvac_1r1c": _hvac_1r1c,
    "hp_ruhnau": _hp_ruhnau,
    "dhw_jordanvajen": _dhw_jordanvajen,
    "pv_pvlib": _pv_pvlib,
}


def _generate(objects: pd.DataFrame, data: dict, workers: int, batch_size, reverse: bool):
    gen = Generator(raise_on_error=True)
    gen.add_objects(objects.iloc[::-1] if reverse else objects)
    return gen.generate(data, workers=workers, batch_size=batch_size, show_progress=False)


@pytest.mark.parametrize("case", CASES)
def test_timeseries_are_bit_identical_however_the_objects_are_run(case):
    objects, data = CASES[case]()
    assert len(objects) > 1

    ref_summary, ref_timeseries = _generate(objects, data, *REFERENCE)
    assert set(ref_timeseries) == set(objects[O.ID])

    for label, variant in VARIANTS.items():
        summary, timeseries = _generate(objects, data, *variant)
        assert set(timeseries) == set(ref_timeseries), label
        for obj_id, by_type in ref_timeseries.items():
            assert set(timeseries[obj_id]) == set(by_type), f"{label}, object {obj_id}"
            for ts_type, expected in by_type.items():
                pd.testing.assert_frame_equal(
                    timeseries[obj_id][ts_type],
                    expected,
                    check_exact=True,
                    obj=f"{case} ({label}), object {obj_id}, {ts_type}",
                )
        pd.testing.assert_frame_equal(summary.loc[ref_summary.index], ref_summary, check_exact=True, obj=label)


def _in_fresh_process(probe):
    # Isolate the worker pool and its startup environment from other tests.
    run = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({__file__!r})[{probe.__name__!r}]()"],
        env={**os.environ, "ENTISE_ACCELERATOR": "auto"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr


class _HvacAcceleratorProbe(Generator):
    def _process_object(self, obj, data):
        resolve = perf.get_accelerator
        resolved = []

        def record_accelerator():
            mode = resolve()
            resolved.append(mode)
            return mode

        with patch.object(perf, "get_accelerator", record_accelerator):
            result = super()._process_object(obj, data)
        (mode,) = resolved
        result[K.SUMMARY].update(accelerator=mode, worker_pid=os.getpid())
        return result


def _check_reused_hvac_workers():
    objects, data = _hvac_1r1c()
    data["weather"] = data["weather"].iloc[:48]
    generator = _HvacAcceleratorProbe(raise_on_error=True)
    generator.add_objects([{**objects.iloc[i % len(objects)].to_dict(), O.ID: i} for i in range(4)])
    worker_pids = set()

    for mode, expected, progress in (("auto", "numba", False), ("none", "none", True), ("auto", "numba", False)):
        perf.set_accelerator(mode)
        summary, timeseries = generator.generate(data, workers=2, batch_size=1, show_progress=progress)
        assert set(timeseries) == set(range(4))
        assert all(len(by_type[Types.HVAC]) == 48 for by_type in timeseries.values())
        pids = set(summary["worker_pid"])
        assert os.getpid() not in pids
        if worker_pids:
            assert pids & worker_pids, "expected a worker from the previous run"
        worker_pids = pids
        assert set(summary["accelerator"]) == {expected}, summary["accelerator"].tolist()


def test_accelerator_mode_reaches_reused_hvac_workers():
    pytest.importorskip("numba")
    _in_fresh_process(_check_reused_hvac_workers)


class _MissingNumbaProbe(Generator):
    def _process_object(self, obj, data):
        with patch.object(perf, "_numba_available", return_value=False):
            return {O.ID: obj[O.ID], K.SUMMARY: {"accelerator": perf.get_accelerator()}, K.TIMESERIES: {}}


def _check_missing_numba_in_workers():
    objects = [{O.ID: 1}, {O.ID: 2}]
    generator = _MissingNumbaProbe()
    generator.add_objects(objects)
    args = {"workers": 2, "show_progress": False}

    perf.set_accelerator("auto")
    summary, _ = generator.generate({}, **args)
    assert set(summary["accelerator"]) == {"none"}

    perf.set_accelerator("numba")
    summary, timeseries = generator.generate({}, **args)
    message = "ENTISE_ACCELERATOR=numba was requested but numba is not installed."
    assert len(summary) == len(objects) and "error" in summary, summary
    assert summary["error"].str.contains(message, regex=False).all()
    assert timeseries == {1: {}, 2: {}}

    generator.raise_on_error = True
    with pytest.raises(ImportError, match="numba was requested"):
        generator.generate({}, **args)

    # The runner must defer resolution until a strategy actually needs it.
    unaccelerated = Generator(raise_on_error=True)
    unaccelerated.add_objects(objects)
    with patch.object(perf, "_numba_available", side_effect=AssertionError("unexpected accelerator resolution")):
        _, timeseries = unaccelerated.generate({}, **args)
    assert timeseries == {1: {}, 2: {}}


def test_required_numba_errors_in_workers_without_numba():
    _in_fresh_process(_check_missing_numba_in_workers)
