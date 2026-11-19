"""Random number generators for methods that draw random numbers.

A method gets its generator from :func:`method_rng`:

- An object ``seed`` wins. The generator is ``numpy.random.default_rng(seed)``,
  so an object with a seed gives the same result with or without a run seed.
- Otherwise a run seed, ``Generator.generate(..., seed=...)``, is combined with
  the object's id and the time series type:
  ``SeedSequence([run_seed, stable_hash(id), stable_hash(ts_type)])``. Each
  object and type draws its own stream, which does not depend on the order of
  the objects, the number of workers, the batch size or the process.
- Without either seed the generator draws fresh entropy, so results differ
  between runs.

The run seed travels to the methods in the run's :class:`~entise.core._time.RunTime`.

This module imports only numpy, the standard library and EnTiSe's core, so a
method that uses it still imports only its own extra.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

from entise.core._time import RUN_TIME_KEY


def stable_hash(value: Any) -> int:
    """Return a 128-bit hash of ``str(value)`` that is the same in every process.

    Python's ``hash()`` of a string changes from one process to the next, so it
    cannot place an object's random stream.
    """
    return int.from_bytes(hashlib.blake2b(str(value).encode("utf-8"), digest_size=16).digest(), "little")


def check_run_seed(seed: Any) -> int | None:
    """Return the run seed as an ``int``, or ``None`` if there is none.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` is negative.
    """
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"The run seed must be a non-negative integer, not {seed!r}.")
    if seed < 0:
        raise ValueError(f"The run seed must be a non-negative integer, not {seed}.")
    return int(seed)


def _object_seed(seed: Any) -> Any:
    """Return an object seed for numpy, or ``None`` if the object has none."""
    if seed is None or (isinstance(seed, (float, np.floating)) and math.isnan(seed)):
        return None
    # An objects table read by pandas stores a seed as a float (1.0) when other objects leave it empty.
    if isinstance(seed, (float, np.floating)) and float(seed).is_integer():
        return int(seed)
    return seed


def method_rng(data: Mapping[str, Any], obj_id: Any, ts_type: str, seed: Any = None) -> np.random.Generator:
    """Return the random number generator for one object and time series type.

    Args:
        data: The data passed to the method. A run seed is read from its run time.
        obj_id: The object's id.
        ts_type: The time series type the method generates.
        seed: The object's own seed, if it has one. It takes precedence over the run seed.

    Returns:
        A numpy generator; see the module docstring for how it is seeded.
    """
    seed = _object_seed(seed)
    if seed is not None:
        return np.random.default_rng(seed)
    run_seed = getattr(data.get(RUN_TIME_KEY), "seed", None)
    if run_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([run_seed, stable_hash(obj_id), stable_hash(ts_type)]))
