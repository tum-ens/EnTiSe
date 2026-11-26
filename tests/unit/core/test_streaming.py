"""Chunked generation, the time-series sink interface and the order of the summary."""

import gc
import inspect
import weakref

import numpy as np
import pandas as pd
import pytest

import entise
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method
from entise.core.generator import Generator
from entise.core.storage import TimeseriesStorage

STEPS = 4
# Weak references to the frames of an earlier chunk, and whether each was freed when the next chunk ran.
WATCHED: list = []
FREED: list = []


class _StreamProbe(Method):
    """Returns the object's ``value`` as a time series; fails for objects with ``fail`` set."""

    types = [Types.HVAC, Types.DHW]
    name = "stream_probe"
    required_keys = []
    optional_keys = ["value", "fail", "quiet"]
    output_timeseries = {}

    def generate(self, obj=None, data=None, results=None, ts_type=Types.HVAC, **inputs):
        obj = {**(obj or {}), **inputs}
        if WATCHED:
            gc.collect()
            FREED.extend(ref() is None for ref in WATCHED)
            WATCHED.clear()
        if obj.get("fail") is True:
            raise RuntimeError(f"object {obj[O.ID]} failed")
        value = float(obj.get("value", 1.0))
        series = pd.DataFrame({f"{ts_type}[W]": np.full(STEPS, value)})
        summary = {} if obj.get("quiet") is True else {f"{ts_type}:total[Wh]": value * STEPS}
        return {K.SUMMARY: summary, K.TIMESERIES: series}


def _objects(n: int, **extra) -> list:
    return [
        {O.ID: f"b{i}", Types.HVAC: "stream_probe", Types.DHW: "stream_probe", "value": i, **extra} for i in range(n)
    ]


def _generator(objects, raise_on_error=False) -> Generator:
    gen = Generator(raise_on_error=raise_on_error)
    gen.add_objects(objects)
    return gen


class _RecordingSink(TimeseriesStorage):
    """Records each call; ``fail_on`` names the call that raises (``write_batch:2`` for the second chunk)."""

    def __init__(self, chunk_size=2, fail_on=None, close_fails=False):
        self.chunk_size = chunk_size
        self.fail_on = fail_on
        self.close_fails = close_fails
        self.calls = []
        self.written = []

    def _call(self, name):
        self.calls.append(name)
        count = self.calls.count(name)
        if self.fail_on in (name, f"{name}:{count}"):
            raise OSError(f"{name} failed")

    def setup(self):
        self._call("setup")

    def write_batch(self, timeseries):
        self._call("write_batch")
        self.written.append(list(timeseries))

    def finalize(self):
        self._call("finalize")

    def close(self):
        self.calls.append("close")
        if self.close_fails:
            raise OSError("close failed")


# -- iter_generate ----------------------------------------------------------------------------------------------------


def test_chunks_hold_whole_objects_in_input_order():
    chunks = list(_generator(_objects(5)).iter_generate({}, chunk_size=2, workers=1, show_progress=False))

    assert [list(ts) for _, ts in chunks] == [["b0", "b1"], ["b2", "b3"], ["b4"]]
    for summary, timeseries in chunks:
        assert list(summary.index) == list(timeseries)
        for by_type in timeseries.values():
            assert set(by_type) == {Types.HVAC, Types.DHW}


def test_concatenated_chunks_equal_one_run():
    objects = _objects(5)
    objects[1]["fail"] = True
    objects[3]["quiet"] = True
    ref_summary, ref_timeseries = _generator(objects).generate({}, workers=1, show_progress=False)

    chunks = list(_generator(objects).iter_generate({}, chunk_size=2, workers=1, show_progress=False))

    pd.testing.assert_frame_equal(pd.concat([s for s, _ in chunks]), ref_summary, check_exact=True)
    timeseries = {k: v for _, ts in chunks for k, v in ts.items()}
    assert list(timeseries) == list(ref_timeseries)
    for obj_id, by_type in ref_timeseries.items():
        for ts_type, frame in by_type.items():
            pd.testing.assert_frame_equal(timeseries[obj_id][ts_type], frame, check_exact=True)


def _watch(timeseries):
    FREED.clear()
    WATCHED[:] = [weakref.ref(frame) for by_type in timeseries.values() for frame in by_type.values()]


def test_a_chunk_the_caller_dropped_is_freed_before_the_next_is_computed():
    chunks = _generator(_objects(2)).iter_generate({}, chunk_size=1, workers=1, show_progress=False)

    summary, timeseries = next(chunks)
    _watch(timeseries)
    del summary, timeseries
    next(chunks)

    assert FREED == [True, True]


def test_a_sink_chunk_is_freed_before_the_next_is_computed():
    class WatchingSink(_RecordingSink):
        def write_batch(self, timeseries):
            super().write_batch(timeseries)
            if len(self.written) == 1:
                _watch(timeseries)

    _generator(_objects(2)).generate({}, workers=1, show_progress=False, storage=WatchingSink(chunk_size=1))

    assert FREED == [True, True]


def test_a_failing_object_keeps_its_error_and_spares_its_chunk():
    objects = _objects(4)
    objects[1]["fail"] = True

    (summary, timeseries), _ = _generator(objects).iter_generate({}, chunk_size=2, workers=1, show_progress=False)

    assert summary.loc["b1", "error"] == "object b1 failed"
    assert timeseries["b1"] == {}
    assert set(timeseries["b0"]) == {Types.HVAC, Types.DHW}
    assert pd.isna(summary.loc["b0", "error"])


def test_a_failing_object_raises_with_raise_on_error():
    objects = _objects(4)
    objects[3]["fail"] = True
    chunks = _generator(objects, raise_on_error=True).iter_generate({}, chunk_size=2, workers=1, show_progress=False)

    summary, _ = next(chunks)
    assert list(summary.index) == ["b0", "b1"]
    with pytest.raises(RuntimeError, match="object b3 failed"):
        next(chunks)


@pytest.mark.parametrize(
    "objects, kwargs, error, match",
    [
        ([], {}, ValueError, "No objects"),
        (_objects(2), {"seed": -1}, ValueError, "seed"),
        (_objects(2), {"chunk_size": 0}, ValueError, "chunk_size"),
        (_objects(2), {"chunk_size": 2.5}, ValueError, "chunk_size"),
        (_objects(2) + _objects(1), {}, ValueError, "b0"),
    ],
    ids=["no objects", "negative seed", "empty chunks", "fractional chunks", "duplicate ids"],
)
def test_bad_arguments_raise_at_the_call(objects, kwargs, error, match):
    gen = _generator(objects) if objects else Generator()

    with pytest.raises(error, match=match):
        gen.iter_generate({}, workers=1, show_progress=False, **kwargs)


def test_iter_generate_defaults():
    parameters = inspect.signature(Generator.iter_generate).parameters

    assert parameters["chunk_size"].default == 500
    generate = inspect.signature(Generator.generate).parameters
    for name, parameter in generate.items():
        if name not in ("self", "storage"):
            assert parameters[name].default == parameter.default, name


# -- generate(storage=...) ---------------------------------------------------------------------------------------------


def test_generate_keeps_its_defaults():
    parameters = inspect.signature(Generator.generate).parameters

    assert {name: p.default for name, p in parameters.items() if name not in ("self", "data")} == {
        "workers": -1,
        "batch_size": None,
        "pre_dispatch": "2*n_jobs",
        "backend": "loky",
        "show_progress": True,
        "utc": False,
        "seed": None,
        "storage": None,
    }


def test_a_sink_gets_each_chunk_once_and_the_summary_is_returned():
    objects = _objects(5)
    objects[2]["fail"] = True
    ref_summary, _ = _generator(objects).generate({}, workers=1, show_progress=False)
    sink = _RecordingSink(chunk_size=2)

    summary, timeseries = _generator(objects).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.calls == ["setup", "write_batch", "write_batch", "write_batch", "finalize"]
    assert sink.written == [["b0", "b1"], ["b2", "b3"], ["b4"]]
    assert timeseries == {}
    pd.testing.assert_frame_equal(summary, ref_summary, check_exact=True)


def test_a_sink_without_chunk_size_uses_the_default(monkeypatch):
    sink = _RecordingSink()
    del sink.chunk_size
    assert TimeseriesStorage.chunk_size == 500

    _generator(_objects(3)).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.written == [["b0", "b1", "b2"]]


def test_a_failing_write_closes_the_sink_and_keeps_earlier_chunks():
    sink = _RecordingSink(chunk_size=2, fail_on="write_batch:2")

    with pytest.raises(OSError, match="write_batch failed"):
        _generator(_objects(5)).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.calls == ["setup", "write_batch", "write_batch", "close"]
    assert sink.written == [["b0", "b1"]]


def test_a_failing_close_does_not_hide_the_first_error(caplog):
    sink = _RecordingSink(chunk_size=2, fail_on="write_batch:1", close_fails=True)

    with pytest.raises(OSError, match="write_batch failed"):
        _generator(_objects(3)).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.calls == ["setup", "write_batch", "close"]
    assert "close failed" in caplog.text


@pytest.mark.parametrize(
    "fail_on, calls",
    [
        ("setup", ["setup"]),
        ("finalize", ["setup", "write_batch", "write_batch", "finalize", "close"]),
    ],
)
def test_a_failing_finalize_closes_the_sink_and_a_failing_setup_does_not(fail_on, calls):
    sink = _RecordingSink(chunk_size=2, fail_on=fail_on)

    with pytest.raises(OSError, match=f"{fail_on} failed"):
        _generator(_objects(3)).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.calls == calls


def test_a_failing_object_closes_the_sink_with_raise_on_error():
    objects = _objects(4)
    objects[3]["fail"] = True
    sink = _RecordingSink(chunk_size=2)

    with pytest.raises(RuntimeError, match="object b3 failed"):
        _generator(objects, raise_on_error=True).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.calls == ["setup", "write_batch", "close"]
    assert sink.written == [["b0", "b1"]]


def test_bad_arguments_raise_before_the_sink_is_set_up():
    sink = _RecordingSink()

    with pytest.raises(ValueError, match="b0"):
        _generator(_objects(2) + _objects(1)).generate({}, workers=1, show_progress=False, storage=sink)

    assert sink.calls == []


# -- object ids --------------------------------------------------------------------------------------------------------


def _run(gen: Generator, how: str, **kwargs):
    if how == "iter_generate":
        return gen.iter_generate({}, workers=1, show_progress=False, **kwargs)
    return gen.generate({}, workers=1, show_progress=False, **kwargs)


@pytest.mark.parametrize("how", ["generate", "iter_generate", "storage"])
def test_repeated_ids_raise_before_any_object_is_computed(how, monkeypatch):
    computed = []
    monkeypatch.setattr(_StreamProbe, "generate", lambda self, obj=None, *args, **kwargs: computed.append(obj))
    sink = _RecordingSink()
    kwargs = {"storage": sink} if how == "storage" else {}

    with pytest.raises(ValueError, match=r"^Object ids must be unique; these repeat: 'b0', 'b1'\.$"):
        _run(_generator(_objects(3) + _objects(2)), how, **kwargs)

    assert computed == []
    assert sink.calls == []


def test_more_than_ten_repeated_ids_are_counted():
    with pytest.raises(ValueError, match=r"'b9' and 2 more\.$"):
        _generator(_objects(12) + _objects(12)).generate({}, workers=1, show_progress=False)


def test_unique_ids_keep_every_object():
    summary, timeseries = _generator(_objects(3)).generate({}, workers=1, show_progress=False)

    assert list(summary.index) == ["b0", "b1", "b2"]
    assert list(timeseries) == ["b0", "b1", "b2"]


# -- TimeseriesStorage -------------------------------------------------------------------------------------------------


def test_storage_needs_setup_write_batch_and_finalize():
    class Partial(TimeseriesStorage):
        def setup(self):
            pass

        def write_batch(self, timeseries):
            pass

    with pytest.raises(TypeError, match="finalize"):
        Partial()
    assert entise.TimeseriesStorage is TimeseriesStorage


def test_storage_as_context_manager_finalizes_or_closes():
    with _RecordingSink() as sink:
        sink.write_batch({})
    assert sink.calls == ["setup", "write_batch", "finalize"]

    with pytest.raises(KeyError):
        with _RecordingSink() as sink:
            raise KeyError("stop")
    assert sink.calls == ["setup", "close"]
