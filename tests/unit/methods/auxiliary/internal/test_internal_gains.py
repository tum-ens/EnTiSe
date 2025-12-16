import numpy as np
import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.methods.auxiliary.internal.selector import InternalGains


# Fixtures for minimal data setup
@pytest.fixture
def minimal_weather():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame(index=index)


@pytest.fixture
def timeseries_with_column():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    return pd.DataFrame({"object_1": np.arange(24)}, index=index)


# Tests
def test_internal_inactive_strategy(minimal_weather):
    obj = {}
    data = {O.WEATHER: minimal_weather}
    ig = InternalGains()

    result = ig.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.GAINS_INTERNAL in result.columns
    assert (result[O.GAINS_INTERNAL] == 0).all()


def test_internal_constant_strategy(minimal_weather):
    obj = {O.GAINS_INTERNAL: 100}
    data = {O.WEATHER: minimal_weather}
    ig = InternalGains()

    result = ig.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.GAINS_INTERNAL in result.columns
    assert (result[O.GAINS_INTERNAL] == 100).all()


def test_internal_timeseries_strategy(minimal_weather, timeseries_with_column):
    obj = {O.ID: "object_1", O.GAINS_INTERNAL_COL: "object_1", O.GAINS_INTERNAL: O.GAINS_INTERNAL}
    data = {
        O.GAINS_INTERNAL: timeseries_with_column,
        O.WEATHER: minimal_weather,
    }
    ig = InternalGains()

    result = ig.generate(obj, data)

    assert isinstance(result, pd.DataFrame)
    assert O.GAINS_INTERNAL in result.columns
    assert np.allclose(result[O.GAINS_INTERNAL], np.arange(24))
