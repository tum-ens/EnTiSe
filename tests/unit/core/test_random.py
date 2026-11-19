"""Random numbers of a method come from the object seed, or from the run seed, the object and the type."""

import numpy as np
import pytest

from entise.core import _random as R
from entise.core._time import RUN_TIME_KEY, RunTime
from entise.core.generator import Generator


def _draws(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2**32, size=8)


def _run(seed) -> dict:
    return {RUN_TIME_KEY: RunTime({}, seed=seed)}


def test_stable_hash_does_not_depend_on_the_process():
    # Fixed values: Python's hash() of a string changes from one process to the next.
    assert R.stable_hash("dhw") == R.stable_hash("dhw")
    assert R.stable_hash(1) == R.stable_hash("1")
    assert R.stable_hash("dhw") == 0x54D70E50D9AE0CECE3035451831B0A0E


def test_run_seed_gives_the_same_draws_for_an_object_and_type():
    first = _draws(R.method_rng(_run(11), 1, "dhw"))
    again = _draws(R.method_rng(_run(11), 1, "dhw"))

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, _draws(R.method_rng(_run(12), 1, "dhw")))
    assert not np.array_equal(first, _draws(R.method_rng(_run(11), 2, "dhw")))
    assert not np.array_equal(first, _draws(R.method_rng(_run(11), 1, "electricity")))


def test_object_seed_wins_and_keeps_its_draws():
    expected = _draws(np.random.default_rng(7))

    for data in ({}, _run(11), _run(12)):
        np.testing.assert_array_equal(_draws(R.method_rng(data, 1, "dhw", seed=7)), expected)
    # An objects table stores the seed as a float when another object leaves it empty.
    np.testing.assert_array_equal(_draws(R.method_rng(_run(11), 1, "dhw", seed=7.0)), expected)
    np.testing.assert_array_equal(
        _draws(R.method_rng(_run(11), 1, "dhw", seed=float("nan"))), _draws(R.method_rng(_run(11), 1, "dhw"))
    )


def test_without_any_seed_draws_differ():
    for data in ({}, _run(None)):
        assert not np.array_equal(_draws(R.method_rng(data, 1, "dhw")), _draws(R.method_rng(data, 1, "dhw")))


@pytest.mark.parametrize("seed", [-1, 1.5, True, "1"])
def test_generator_rejects_a_run_seed_that_is_not_a_non_negative_integer(seed):
    gen = Generator()
    gen.add_objects({"id": 1})

    with pytest.raises((TypeError, ValueError), match="seed"):
        gen.generate({}, workers=1, show_progress=False, seed=seed)


def test_run_seed_accepts_numpy_integers():
    assert R.check_run_seed(np.int64(3)) == 3
    assert R.check_run_seed(None) is None
