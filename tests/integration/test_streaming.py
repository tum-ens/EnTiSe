"""Generating objects in chunks must give the same results as generating them at once.

``Generator.iter_generate`` hands the caller one chunk of objects at a time. Each object is
computed exactly as ``generate`` computes it, so the chunk size, the number of workers and the
batch size must not change a single value, and every object must arrive whole, in input order.
"""

import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.core.generator import Generator
from entise.core.storage import TimeseriesStorage
from tests.integration.test_reproducibility import CASES as REPRODUCIBILITY_CASES

CASES = {
    name: REPRODUCIBILITY_CASES[name]
    for name in ("hvac_1r1c", "hvac_1r1c_typed", "hp_ruhnau", "dhw_jordanvajen", "pv_pvlib")
}

# How the chunks are computed: (workers, batch_size)
RUNS = {
    "sequential": (1, None),
    "two workers, even batches": (2, None),
    "two workers, one object per batch": (2, 1),
}


def _generator(objects: pd.DataFrame) -> Generator:
    gen = Generator(raise_on_error=True)
    gen.add_objects(objects)
    return gen


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    objects, data = CASES[request.param]()
    summary, timeseries = _generator(objects).generate(data, workers=1, show_progress=False)
    return request.param, objects, data, summary, timeseries


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("chunk_size", [1, 3, None], ids=["chunk 1", "chunk 3", "chunk n"])
def test_chunks_are_bit_identical_to_one_run(case, run, chunk_size):
    name, objects, data, ref_summary, ref_timeseries = case
    workers, batch_size = RUNS[run]
    size = chunk_size or len(objects)
    label = f"{name} ({run}, chunk size {size})"

    chunks = list(
        _generator(objects).iter_generate(
            data, chunk_size=size, workers=workers, batch_size=batch_size, show_progress=False
        )
    )

    ids = list(objects[O.ID])
    assert [list(timeseries) for _, timeseries in chunks] == [ids[i : i + size] for i in range(0, len(ids), size)]
    for summary, timeseries in chunks:
        assert list(summary.index) == list(timeseries), label
        for obj_id, by_type in timeseries.items():
            assert set(by_type) == set(ref_timeseries[obj_id]), f"{label}, object {obj_id}"
            for ts_type, frame in by_type.items():
                pd.testing.assert_frame_equal(
                    frame,
                    ref_timeseries[obj_id][ts_type],
                    check_exact=True,
                    obj=f"{label}, object {obj_id}, {ts_type}",
                )

    summary = pd.concat([summary for summary, _ in chunks])
    pd.testing.assert_frame_equal(summary, ref_summary, check_exact=True, obj=label)


class _MemorySink(TimeseriesStorage):
    """Keeps every chunk it receives."""

    chunk_size = 2

    def __init__(self):
        self.chunks = []

    def setup(self):
        pass

    def write_batch(self, timeseries):
        self.chunks.append(timeseries)

    def finalize(self):
        pass


@pytest.mark.parametrize("run", RUNS)
def test_a_sink_receives_the_same_results_as_one_run(case, run):
    name, objects, data, ref_summary, ref_timeseries = case
    workers, batch_size = RUNS[run]
    sink = _MemorySink()

    summary, timeseries = _generator(objects).generate(
        data, workers=workers, batch_size=batch_size, show_progress=False, storage=sink
    )

    assert timeseries == {}
    pd.testing.assert_frame_equal(summary, ref_summary, check_exact=True, obj=name)
    received = {}
    for chunk in sink.chunks:
        assert len(chunk) <= sink.chunk_size
        assert not set(chunk) & set(received), "an object arrived in two chunks"
        received.update(chunk)
    assert list(received) == list(ref_timeseries)
    for obj_id, by_type in ref_timeseries.items():
        assert set(received[obj_id]) == set(by_type)
        for ts_type, expected in by_type.items():
            pd.testing.assert_frame_equal(received[obj_id][ts_type], expected, check_exact=True)
