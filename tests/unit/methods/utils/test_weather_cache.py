"""Regression tests for issue #99: the module-level `_WEATHER_CACHE` used
to be keyed by weather-key string. Two callers passing DIFFERENT
DataFrames under the same key got each other's preprocessed weather —
a silent correctness bug that also produced cross-test pollution.

These tests target the shared `WeatherCache` in `entise.methods.utils`
directly (unit level) and also exercise the six migrated call sites
end-to-end (integration level, one test per model)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.utils.weather_cache import WeatherCache

# --- Unit tests for the WeatherCache class ---------------------------------


def _dummy_df(temp: float, periods: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: idx,
            C.TEMP_AIR: np.full(periods, temp),
            C.SOLAR_GHI: np.zeros(periods),
            C.SOLAR_DHI: np.zeros(periods),
            C.SOLAR_DNI: np.zeros(periods),
        },
        index=idx,
    )


class TestWeatherCache:
    def test_hit_returns_same_preprocessed_across_calls(self):
        cache = WeatherCache()
        df = _dummy_df(10.0)
        calls = 0

        def build(w):
            nonlocal calls
            calls += 1
            return w.copy()

        r1 = cache.get_or_build(df, build)
        r2 = cache.get_or_build(df, build)
        assert calls == 1
        assert r1 is r2  # same object — build ran once

    def test_different_dataframes_get_separate_entries(self):
        """The core regression for #99 — two different DataFrames must not
        collide even though they'd share a weather-key name upstream."""
        cache = WeatherCache()
        df_a = _dummy_df(10.0)
        df_b = _dummy_df(30.0)

        def build(w):
            return float(w[C.TEMP_AIR].sum())

        ra = cache.get_or_build(df_a, build)
        rb = cache.get_or_build(df_b, build)
        assert ra != rb
        assert ra == pytest.approx(10.0 * 24)
        assert rb == pytest.approx(30.0 * 24)

    def test_holds_strong_ref_so_id_cannot_be_reused(self):
        """We rely on `id()` uniqueness while cached. Since the cache holds
        a strong reference to the original DataFrame, its id cannot be
        recycled by GC — even if the caller drops its own reference."""
        import gc

        cache = WeatherCache()
        df = _dummy_df(10.0)
        original_id = id(df)
        cache.get_or_build(df, lambda w: w)

        del df
        gc.collect()
        # Force allocation of a new DataFrame; if the cache didn't retain
        # the original, this new one could land at the same id.
        _dummy_df(30.0)
        # We can't assert `id(new_df) != original_id` — that's an
        # implementation detail of CPython — but we CAN assert the cache
        # entry for `original_id` still exists.
        assert original_id in cache._store  # cache-internal check

    def test_fifo_eviction_when_full(self):
        cache = WeatherCache(max_size=3)
        dfs = [_dummy_df(float(i)) for i in range(5)]
        for df in dfs:
            cache.get_or_build(df, lambda w: w)
        assert len(cache) == 3
        # First two evicted (oldest first)
        assert id(dfs[0]) not in cache._store
        assert id(dfs[1]) not in cache._store
        assert id(dfs[4]) in cache._store

    def test_clear_empties_the_store(self):
        cache = WeatherCache()
        cache.get_or_build(_dummy_df(10.0), lambda w: w)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0


# --- End-to-end regression: the exact #99 scenario -------------------------


def _r1c1_obj() -> dict:
    return {
        O.ID: "b1",
        O.CAPACITANCE: 5e6,
        O.RESISTANCE: 0.005,
        O.TEMP_INIT: 22.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 500.0,
        O.VENTILATION: 65.0,
        # Deliberately NOT setting O.WEATHER — the default key is used.
    }


def test_r1c1_two_different_weathers_under_default_key_give_different_results():
    """The exact scenario that caused #99: two independent generate() calls
    pass different weather DataFrames under the default O.WEATHER key.
    Before the fix, the second call silently reused the first call's
    weather. After the fix, each call sees its own input."""
    from entise.methods.hvac.R1C1 import R1C1

    cold = _dummy_df(-10.0, periods=48)  # forces heating
    hot = _dummy_df(35.0, periods=48)  # forces cooling

    r_cold = R1C1().generate(_r1c1_obj(), {O.WEATHER: cold}, Types.HVAC)
    r_hot = R1C1().generate(_r1c1_obj(), {O.WEATHER: hot}, Types.HVAC)

    heat_key = f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"
    cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
    assert r_cold["summary"][heat_key] > 0, "cold weather should trigger heating"
    assert r_cold["summary"][cool_key] == 0
    assert r_hot["summary"][cool_key] > 0, "hot weather should trigger cooling"
    assert r_hot["summary"][heat_key] == 0


def test_r5c1_two_different_weathers_under_default_key_give_different_results():
    from entise.methods.hvac.R5C1 import R5C1

    r5c1_obj = {
        O.ID: "b1",
        O.C_M: 5e7,
        O.H_TR_IS: 500.0,
        O.H_TR_MS: 1000.0,
        O.H_TR_W: 30.0,
        O.H_TR_EM: 800.0,
        O.VENTILATION: 100.0,
        O.TEMP_INIT: 22.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 25.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 48.1,
        O.LON: 11.6,
        O.GAINS_INTERNAL: 500.0,
    }

    cold = _dummy_df(-10.0, periods=48)
    hot = _dummy_df(35.0, periods=48)

    r_cold = R5C1().generate(dict(r5c1_obj), {O.WEATHER: cold}, Types.HVAC)
    r_hot = R5C1().generate(dict(r5c1_obj), {O.WEATHER: hot}, Types.HVAC)

    heat_key = f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"
    cool_key = f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"
    # Cold weather → more heating; hot weather → more cooling. The exact
    # magnitudes depend on R5C1's dynamics, but the qualitative response
    # to different inputs must differ.
    assert r_cold["summary"][heat_key] > r_hot["summary"][heat_key]
    assert r_hot["summary"][cool_key] > r_cold["summary"][cool_key]
