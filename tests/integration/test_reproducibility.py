"""Generating the same object again must give bit-identical time series.

The result of an object may not depend on the order of the objects, on the number of
workers, on the batch size, or on the process that happens to calculate it. Methods that
draw random numbers are reproducible with an object ``seed`` or a run ``seed``.
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
from entise.methods.dhw.jordan_vajen import JordanVajen

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, os.pardir, os.pardir, "examples")
HOURS = 14 * 24
# The first row of a window that spans the change to summer time on 27 March 2022.
CLOCK_CHANGE = 1990

# How the objects are run: (workers, batch_size, reverse the object order)
REFERENCE = (1, None, False)
VARIANTS = {
    "reversed order": (1, None, True),
    "two workers, one object per batch": (2, 1, False),
    "two workers, even batches, reversed order": (2, None, True),
}


def _weather(first: int = 0) -> pd.DataFrame:
    weather = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv"), parse_dates=True)
    return weather.iloc[first : first + HOURS].reset_index(drop=True)


def _example(name: str, first: int = 0):
    """The objects and tables of an example, on a two-week weather window."""
    folder = os.path.join(EXAMPLES, name)
    objects = pd.read_csv(os.path.join(folder, "objects.csv"))
    data = {"weather": _weather(first)}
    tables = os.path.join(folder, "data")
    if os.path.isdir(tables):
        for file in sorted(os.listdir(tables)):
            data[os.path.splitext(file)[0]] = pd.read_csv(os.path.join(tables, file))
    return objects, data


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


def _hvac_5r1c_across_the_clock_change():
    # 5R1C and 7R2C share the solar position and irradiance caches with 1R1C.
    return _example("hvac_5r1c", CLOCK_CHANGE)


def _hvac_7r2c():
    return _example("hvac_7r2c")


def _electricity_demandlib_across_the_clock_change():
    pytest.importorskip("demandlib")
    return _example("electricity_demandlib", CLOCK_CHANGE)


CASES = {
    "hvac_1r1c": _hvac_1r1c,
    "hvac_5r1c across the clock change": _hvac_5r1c_across_the_clock_change,
    "hvac_7r2c": _hvac_7r2c,
    "hp_ruhnau": _hp_ruhnau,
    "dhw_jordanvajen": _dhw_jordanvajen,
    "pv_pvlib": _pv_pvlib,
    "electricity_demandlib across the clock change": _electricity_demandlib_across_the_clock_change,
}


def _generate(objects: pd.DataFrame, data: dict, workers: int, batch_size, reverse: bool, seed=None):
    gen = Generator(raise_on_error=True)
    gen.add_objects(objects.iloc[::-1] if reverse else objects)
    return gen.generate(data, workers=workers, batch_size=batch_size, show_progress=False, seed=seed)


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


# -- run seed ---------------------------------------------------------------------------------------------------------

DHW_VOLUME = f"{Types.DHW}:volume[l]"


def _dhw_without_object_seeds():
    objects = pd.read_csv(os.path.join(HERE, "dhw_jordanvajen", "objects.csv"))
    return objects, {"weather": _weather()}


def _dhw_volumes(timeseries: dict) -> dict:
    return {obj_id: by_type[Types.DHW][DHW_VOLUME] for obj_id, by_type in timeseries.items()}


def test_run_seed_gives_bit_identical_dhw_however_the_objects_are_run():
    objects, data = _dhw_without_object_seeds()
    ref_summary, ref_timeseries = _generate(objects, data, *REFERENCE, seed=11)

    for label, variant in {"same run again": REFERENCE, **VARIANTS}.items():
        summary, timeseries = _generate(objects, data, *variant, seed=11)
        assert set(timeseries) == set(ref_timeseries), label
        for obj_id, by_type in ref_timeseries.items():
            pd.testing.assert_frame_equal(
                timeseries[obj_id][Types.DHW],
                by_type[Types.DHW],
                check_exact=True,
                obj=f"run seed ({label}), object {obj_id}",
            )
        pd.testing.assert_frame_equal(summary.loc[ref_summary.index], ref_summary, check_exact=True, obj=label)


def test_different_run_seeds_give_different_dhw():
    objects, data = _dhw_without_object_seeds()
    _, first = _generate(objects, data, *REFERENCE, seed=11)
    _, second = _generate(objects, data, *REFERENCE, seed=12)

    for obj_id, volume in _dhw_volumes(first).items():
        assert not volume.equals(_dhw_volumes(second)[obj_id]), obj_id


def test_object_seed_overrides_run_seed():
    objects, data = _dhw_without_object_seeds()
    objects[O.SEED] = [7, None, 7]

    _, unseeded_run = _generate(objects, data, *REFERENCE)
    _, first = _generate(objects, data, *REFERENCE, seed=11)
    _, second = _generate(objects, data, *REFERENCE, seed=12)
    unseeded_run, first, second = _dhw_volumes(unseeded_run), _dhw_volumes(first), _dhw_volumes(second)

    for obj_id in (1, 3):
        pd.testing.assert_series_equal(first[obj_id], second[obj_id], check_exact=True)
        pd.testing.assert_series_equal(first[obj_id], unseeded_run[obj_id], check_exact=True)
    # The object without its own seed follows the run seed.
    assert not first[2].equals(second[2])


def test_objects_with_different_ids_draw_different_streams():
    objects, data = _dhw_without_object_seeds()
    twins = pd.concat([objects.iloc[[0]]] * 2, ignore_index=True)
    twins[O.ID] = ["a", "b"]

    _, timeseries = _generate(twins, data, *REFERENCE, seed=11)
    volumes = _dhw_volumes(timeseries)

    assert not volumes["a"].equals(volumes["b"])


def test_direct_call_with_seed_is_reproducible():
    objects, data = _dhw_without_object_seeds()
    obj = objects.iloc[0].to_dict()

    first = JordanVajen().generate(obj, data, seed=5)[K.TIMESERIES]
    again = JordanVajen().generate(obj, data, seed=5)[K.TIMESERIES]
    other = JordanVajen().generate(obj, data, seed=6)[K.TIMESERIES]

    pd.testing.assert_frame_equal(first, again, check_exact=True)
    assert not first[DHW_VOLUME].equals(other[DHW_VOLUME])


# -- accelerator in process workers ------------------------------------------------------------------------------------


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
    # New ids have no rows in the windows table, so the objects run without windows.
    generator.add_objects([{**objects.iloc[i % len(objects)].to_dict(), O.ID: i, O.WINDOWS: None} for i in range(4)])
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
