"""Regression tests for issue #106.

Two defects on the 5R1C / 7R2C ventilation path:

A. ``_prepare_inputs`` multiplied the ventilation auxiliary's output (carrying
   the user's table index) against a split series built on the model index.
   pandas aligned on the *union*, silently producing an over-long, NaN-padded
   frame. The sensible solver truncated it without noticing; the latent-cooling
   post-pass then broadcast the over-long ``H_ve`` against the ``n``-length
   weather arrays and raised
   ``ValueError: operands could not be broadcast together with shapes (n+1,) (n,)``.

B. Neither model could consume a ventilation *timeseries* at all: they never
   declared ``ventilation_column`` (so ``_process_kwargs`` stripped it), filed
   the resolved table under the canonical key instead of the user's key (so the
   auxiliary looked it up and got ``None``), and 7R2C omitted ``O.VENTILATION``
   from ``optional_data`` entirely.
"""

import numpy as np
import pandas as pd
import pytest

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.methods.hvac.R5C1 import R5C1
from entise.methods.hvac.R7C2 import R7C2

N = 48


@pytest.fixture
def humid_weather():
    """Warm, humid weather so the cooling *and* latent paths both engage."""
    index = pd.date_range("2025-07-01", periods=N, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            C.DATETIME: index,
            C.TEMP_AIR: np.full(N, 30.0),
            C.HUMIDITY_REL: np.full(N, 0.75),
            C.SURFACE_AIR_PRESSURE: np.full(N, 101325.0),
            C.SOLAR_GHI: np.full(N, 300.0),
            C.SOLAR_DHI: np.full(N, 80.0),
            C.SOLAR_DNI: np.full(N, 250.0),
        },
        index=index,
    )


@pytest.fixture
def obj_5r1c():
    return {
        O.ID: "test_5r1c",
        O.C_M: 5e7,
        O.H_TR_IS: 500.0,
        O.H_TR_MS: 1000.0,
        O.H_TR_W: 30.0,
        O.H_TR_EM: 800.0,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
    }


@pytest.fixture
def obj_7r2c():
    return {
        O.ID: "test_7r2c",
        O.R_1_AW: 0.002,
        O.C_1_AW: 4e6,
        O.R_1_IW: 0.002,
        O.C_1_IW: 1e7,
        O.R_ALPHA_STAR_IL: 0.0006,
        O.R_ALPHA_STAR_AW: 0.002,
        O.R_ALPHA_STAR_IW: 0.0008,
        O.R_REST_AW: 0.015,
        O.TEMP_INIT: 20.0,
        O.TEMP_MIN: 20.0,
        O.TEMP_MAX: 24.0,
        O.AREA: 150.0,
        O.HEIGHT: 2.7,
        O.LAT: 49.0,
        O.LON: 11.0,
    }


@pytest.fixture
def models(obj_5r1c, obj_7r2c):
    return {"5R1C": (R5C1(), obj_5r1c), "7R2C": (R7C2(), obj_7r2c)}


MODEL_IDS = ["5R1C", "7R2C"]


def _vent_table(index_like, value=90.0, col="typical"):
    return pd.DataFrame({col: np.full(len(index_like), value, dtype=float)}, index=index_like)


# --------------------------------------------------------------------------
# B — a ventilation timeseries is usable at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_ventilation_column_is_declared(models, model_key):
    """`ventilation_column` must survive `_process_kwargs`.

    It is filtered against `required_keys + optional_keys`, so a key the model
    reads but never declares is silently dropped on the keyword API.
    """
    method, _ = models[model_key]
    obj, _ = method._process_kwargs({O.ID: "x"}, {}, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical"})
    assert obj[O.VENTILATION_COL] == "typical"


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_ventilation_from_positional_table(models, model_key, humid_weather):
    """A table read from CSV carries a RangeIndex; align it positionally."""
    method, obj = models[model_key]
    obj = dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical"})
    data = {O.WEATHER: humid_weather, "vent_tbl": _vent_table(range(N))}

    result = method.generate(obj, data, Types.HVAC)

    assert len(result["timeseries"]) == N
    assert result["timeseries"].notna().all().all()


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_ventilation_timeseries_matches_equivalent_scalar(models, model_key, humid_weather):
    """A constant timeseries must reproduce the equivalent scalar exactly."""
    method, obj = models[model_key]
    # `[W/K]` is the spelling `VentilationTimeSeries` reads as a conductance.
    data = {O.WEATHER: humid_weather, "vent_tbl": _vent_table(humid_weather.index, col="typical[W/K]")}

    scalar = method.generate(dict(obj, **{O.VENTILATION: 90.0}), {O.WEATHER: humid_weather}, Types.HVAC)
    series = method.generate(
        dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical[W/K]"}),
        data,
        Types.HVAC,
    )

    pd.testing.assert_frame_equal(scalar["timeseries"], series["timeseries"])


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_ventilation_timeseries_actually_reaches_the_solver(models, model_key, humid_weather):
    """Guard against the fix degrading into a silent zero-ventilation run.

    On this warm weather more outdoor air means more cooling, so a higher
    conductance must raise cooling demand.
    """
    method, obj = models[model_key]
    demands = []
    for value in (20.0, 400.0):
        data = {
            O.WEATHER: humid_weather,
            "vent_tbl": _vent_table(humid_weather.index, value=value, col="typical[W/K]"),
        }
        result = method.generate(
            dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical[W/K]"}),
            data,
            Types.HVAC,
        )
        demands.append(result["summary"][f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]"])

    assert demands[1] > demands[0]


# --------------------------------------------------------------------------
# A — index alignment of the ventilation auxiliary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_longer_ventilation_table_is_aligned_by_label(models, model_key, humid_weather):
    """A table covering one extra trailing timestamp used to yield an n+1 frame.

    The solver truncated it silently and the latent post-pass then broke with
    a broadcast error. The extra timestamp must simply be dropped.
    """
    method, obj = models[model_key]
    long_index = humid_weather.index.append(pd.DatetimeIndex([humid_weather.index[-1] + pd.Timedelta(1, unit="h")]))
    data = {
        O.WEATHER: humid_weather,
        "vent_tbl": _vent_table(long_index, col="typical[W/K]"),
    }
    obj = dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical[W/K]"})

    result = method.generate(obj, data, Types.HVAC)

    ts = result["timeseries"]
    assert len(ts) == N
    assert ts.notna().all().all()
    # Same answer as the equivalent scalar — the extra row changed nothing.
    scalar = method.generate(dict(obj, **{O.VENTILATION: 90.0}), {O.WEATHER: humid_weather}, Types.HVAC)
    pd.testing.assert_frame_equal(scalar["timeseries"], ts)


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_latent_cooling_survives_a_ventilation_timeseries(models, model_key, humid_weather):
    """The latent post-pass was the first consumer to notice the misalignment."""
    method, obj = models[model_key]
    data = {
        O.WEATHER: humid_weather,
        "vent_tbl": _vent_table(humid_weather.index, value=400.0, col="typical[W/K]"),
    }
    obj = dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical[W/K]"})

    ts = method.generate(obj, data, Types.HVAC)["timeseries"]

    latent = ts[f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]"]
    sensible = ts[f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]"]
    total = ts[f"{Types.COOLING}{SEP}{C.LOAD}[W]"]
    assert latent.max() > 0, "humid weather + ventilation must produce a latent load"
    assert (total == sensible + latent).all()


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_ventilation_table_with_a_gap_raises_a_clear_error(models, model_key, humid_weather):
    """A table missing timestamps must name the problem, not fail downstream."""
    method, obj = models[model_key]
    gapped = _vent_table(humid_weather.index, col="typical[W/K]").iloc[2:]
    data = {O.WEATHER: humid_weather, "vent_tbl": gapped}
    obj = dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical[W/K]"})

    with pytest.raises(ValueError, match="does not cover the model index"):
        method.generate(obj, data, Types.HVAC)


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_positional_ventilation_table_of_wrong_length_raises(models, model_key, humid_weather):
    """No index to align on and the wrong length: say so, don't broadcast."""
    method, obj = models[model_key]
    data = {O.WEATHER: humid_weather, "vent_tbl": _vent_table(range(N - 5))}
    obj = dict(obj, **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical"})

    with pytest.raises(ValueError, match="steps but the model index has"):
        method.generate(obj, data, Types.HVAC)


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_scalar_ventilation_unchanged(models, model_key, humid_weather):
    """The scalar path must be untouched by the alignment fix."""
    method, obj = models[model_key]
    result = method.generate(dict(obj, **{O.VENTILATION: 90.0}), {O.WEATHER: humid_weather}, Types.HVAC)
    assert len(result["timeseries"]) == N
    assert result["timeseries"].notna().all().all()


@pytest.mark.parametrize("model_key", MODEL_IDS)
def test_inactive_ventilation_unchanged(models, model_key, humid_weather):
    """`active_ventilation=False` still zeroes the conductance."""
    method, obj = models[model_key]
    obj = dict(
        obj,
        **{O.VENTILATION: "vent_tbl", O.VENTILATION_COL: "typical[W/K]", O.ACTIVE_VENTILATION: False},
    )
    data = {O.WEATHER: humid_weather, "vent_tbl": _vent_table(humid_weather.index, col="typical[W/K]")}

    off = method.generate(obj, data, Types.HVAC)
    zero = method.generate(
        dict(obj, **{O.VENTILATION: 0.0, O.ACTIVE_VENTILATION: True}),
        {O.WEATHER: humid_weather},
        Types.HVAC,
    )

    pd.testing.assert_frame_equal(off["timeseries"], zero["timeseries"])
