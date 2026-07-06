"""Shared, identity-keyed cache for preprocessed weather DataFrames.

Every HVAC / heat method that consumes a weather series needs the same
preprocessing (strip ``@height`` suffixes, parse the DATETIME column,
set the index). Doing that on every call is expensive when many objects
share the same weather DataFrame, so each caller kept a module-level
``_WEATHER_CACHE: dict[str, pd.DataFrame]`` keyed by the weather-key
string.

That was unsafe. Two callers passing different DataFrames under the same
key (e.g. both defaulting to ``O.WEATHER = "weather"``) got each other's
preprocessed weather back — a silent correctness bug that also caused
cross-test pollution (see issue #99).

This module replaces the six ad-hoc caches with one class, keyed by the
**identity** of the input DataFrame. Since the cache also holds a strong
reference to the original DataFrame, its ``id()`` can't be reused via
garbage collection while the entry lives — so ``id()`` is a safe key.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

# Cap prevents unbounded growth if a long-lived process feeds distinct
# DataFrames on every call (e.g. rebuilding from CSV per call). Real
# workflows typically hoist one weather DataFrame and reuse it, so this
# cap should almost never be hit in practice. FIFO eviction — good
# enough given the typical single-DataFrame usage.
_DEFAULT_MAX_SIZE = 128


class WeatherCache:
    """Identity-keyed cache for a per-caller preprocessed weather DataFrame.

    Each HVAC/heat model instantiates its own ``WeatherCache`` because the
    preprocessing differs slightly per model. The class only cares about
    the cache mechanics; the caller supplies its own ``build`` callable.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._max_size = max_size
        # id(original_df) -> (original_df, preprocessed_df)
        # Keeping ``original_df`` alive means Python won't recycle its id
        # to another object while the entry lives — so the id is a safe key.
        self._store: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def get_or_build(
        self,
        weather: pd.DataFrame,
        build: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> pd.DataFrame:
        """Return the preprocessed version of ``weather``, building it once.

        If the same DataFrame object is passed again, the cached
        preprocessed version is returned. If a *different* DataFrame is
        passed (even under the same weather-key name), it gets its own
        cache entry — no cross-contamination.
        """
        key = id(weather)
        cached = self._store.get(key)
        if cached is not None:
            return cached[1]
        preprocessed = build(weather)
        if len(self._store) >= self._max_size:
            # Evict the oldest entry. dict preserves insertion order
            # since Python 3.7, so the first key is the oldest.
            self._store.pop(next(iter(self._store)))
        self._store[key] = (weather, preprocessed)
        return preprocessed

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
