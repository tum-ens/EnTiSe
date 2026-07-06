"""Unit tests for the shared latent-cooling post-pass used by the RC HVAC
models (issue #103).

These tests exercise the pure function ``compute_latent_cooling`` directly on
synthetic arrays so they are decoupled from any particular RC solver. Model
integration is covered by the R1C1 / R5C1 / R7C2 test suites.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from entise.constants import Columns as C
from entise.methods.hvac._latent_cooling import compute_latent_cooling

# --- Helpers -----------------------------------------------------------------


def _weather(index, t_out_c, rh=None, p_pa=None):
    cols = {C.DATETIME: index, C.TEMP_AIR: np.full(len(index), t_out_c, dtype=np.float64)}
    if rh is not None:
        cols[C.HUMIDITY_REL] = np.full(len(index), rh, dtype=np.float64)
    if p_pa is not None:
        cols[C.SURFACE_AIR_PRESSURE] = np.full(len(index), p_pa, dtype=np.float64)
    return pd.DataFrame(cols, index=index)


def _index(n=24):
    return pd.date_range("2025-06-01", periods=n, freq="h", tz="UTC")


# --- Missing weather columns → graceful degradation --------------------------


def test_missing_relative_humidity_returns_zeros_and_warns(caplog):
    """When the weather DataFrame lacks a humidity column, latent must be
    zero everywhere and the helper must emit exactly one warning naming the
    missing column. Silent zero-output is a footgun the user explicitly
    asked us to avoid."""
    idx = _index()
    weather = _weather(idx, t_out_c=30.0, p_pa=101325.0)  # no RH
    p_sens = np.full(24, 500.0, dtype=np.float64)
    p_cap = np.full(24, np.inf, dtype=np.float64)
    h_ve = np.full(24, 100.0, dtype=np.float64)
    gains_lat = np.zeros(24, dtype=np.float64)

    with caplog.at_level(logging.WARNING, logger="entise.methods.hvac._latent_cooling"):
        p_sens_out, p_lat = compute_latent_cooling(
            weather=weather,
            p_cool_sensible=p_sens,
            p_cool_max=p_cap,
            h_ve=h_ve,
            gains_internal_latent=gains_lat,
            temp_max_c=24.0,
            target_humidity_rel=0.5,
        )

    assert np.all(p_lat == 0)
    assert np.allclose(p_sens_out, p_sens)
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert C.HUMIDITY_REL in warnings[0].message


def test_missing_pressure_returns_zeros_and_warns(caplog):
    """Same graceful degradation for missing surface pressure. Symmetric
    with the missing-RH test above: exactly one warning naming the missing
    column so a future refactor that logs twice fails loudly."""
    idx = _index()
    weather = _weather(idx, t_out_c=30.0, rh=0.7)  # no pressure
    p_sens = np.full(24, 500.0, dtype=np.float64)
    p_cap = np.full(24, np.inf, dtype=np.float64)

    with caplog.at_level(logging.WARNING, logger="entise.methods.hvac._latent_cooling"):
        _, p_lat = compute_latent_cooling(
            weather=weather,
            p_cool_sensible=p_sens,
            p_cool_max=p_cap,
            h_ve=np.full(24, 100.0),
            gains_internal_latent=np.zeros(24),
            temp_max_c=24.0,
            target_humidity_rel=0.5,
        )

    assert np.all(p_lat == 0)
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert C.SURFACE_AIR_PRESSURE in warnings[0].message


def test_missing_both_columns_warns_once_per_column(caplog):
    """When both humidity and pressure are missing, both are named exactly
    once so the user can fix them in one go."""
    idx = _index()
    weather = _weather(idx, t_out_c=30.0)  # neither RH nor pressure

    with caplog.at_level(logging.WARNING, logger="entise.methods.hvac._latent_cooling"):
        _, p_lat = compute_latent_cooling(
            weather=weather,
            p_cool_sensible=np.full(24, 500.0),
            p_cool_max=np.full(24, np.inf),
            h_ve=np.full(24, 100.0),
            gains_internal_latent=np.zeros(24),
            temp_max_c=24.0,
            target_humidity_rel=0.5,
        )

    assert np.all(p_lat == 0)
    messages = " ".join(rec.message for rec in caplog.records if rec.levelno == logging.WARNING)
    assert C.HUMIDITY_REL in messages
    assert C.SURFACE_AIR_PRESSURE in messages


# --- Physics happy paths -----------------------------------------------------


def test_dry_outdoor_air_produces_zero_latent():
    """When outdoor absolute humidity is below indoor target and there are
    no internal latent gains, net moisture load is negative. An AC coil
    can't run in reverse — latent must be clipped at 0."""
    idx = _index()
    # 20°C 20% RH → very dry. Indoor target at 24°C 50% RH is much wetter.
    weather = _weather(idx, t_out_c=20.0, rh=0.20, p_pa=101325.0)

    _, p_lat = compute_latent_cooling(
        weather=weather,
        p_cool_sensible=np.full(24, 500.0),  # coil is running
        p_cool_max=np.full(24, np.inf),
        h_ve=np.full(24, 100.0),
        gains_internal_latent=np.zeros(24),
        temp_max_c=24.0,
        target_humidity_rel=0.5,
    )

    assert np.all(p_lat == 0)


def test_coil_off_zeros_latent_even_when_moisture_load_positive():
    """A mild-but-humid day where sensible cooling is zero (T_in stays
    below T_max) must still show zero latent. A standard split AC without
    a dedicated dehum mode doesn't dehumidify when the compressor is off.
    Documented model limitation."""
    idx = _index()
    weather = _weather(idx, t_out_c=25.0, rh=0.85, p_pa=101325.0)  # muggy

    _, p_lat = compute_latent_cooling(
        weather=weather,
        p_cool_sensible=np.zeros(24),  # coil idle
        p_cool_max=np.full(24, np.inf),
        h_ve=np.full(24, 100.0),
        gains_internal_latent=np.full(24, 300.0),  # significant latent gain
        temp_max_c=24.0,
        target_humidity_rel=0.5,
    )

    assert np.all(p_lat == 0)


def test_wet_outdoor_air_produces_expected_latent_load():
    """Muggy outdoor air (30°C, 70% RH) with coil on should produce a
    latent load that matches the analytical formula
        Q_lat[W] = m_dot_air · (ω_out − ω_target) · h_fg
    within tight tolerance. Verifies the physics rather than the plumbing."""
    from entise.methods.utils.psychrometrics import (
        LATENT_HEAT_VAPORISATION as h_fg,
    )
    from entise.methods.utils.psychrometrics import (
        humidity_ratio,
    )

    idx = _index()
    rh_out = 0.70
    t_out = 30.0
    p = 101325.0
    t_max = 24.0
    rh_target = 0.5
    h_ve = 100.0  # W/K
    weather = _weather(idx, t_out_c=t_out, rh=rh_out, p_pa=p)

    _, p_lat = compute_latent_cooling(
        weather=weather,
        p_cool_sensible=np.full(24, 1.0),  # tiny sensible → coil "on"
        p_cool_max=np.full(24, np.inf),
        h_ve=np.full(24, h_ve),
        gains_internal_latent=np.zeros(24),
        temp_max_c=t_max,
        target_humidity_rel=rh_target,
    )

    m_dot = h_ve / 1000.0  # c_p_air = 1000 J/kgK, matches ventilation strategies
    omega_out = humidity_ratio(rh=rh_out, temp_c=t_out, p_pa=p)
    omega_tgt = humidity_ratio(rh=rh_target, temp_c=t_max, p_pa=p)
    expected = m_dot * (omega_out - omega_tgt) * h_fg

    assert np.all(np.isclose(p_lat, expected, rtol=1e-6))
    assert expected > 0


def test_internal_latent_gains_add_to_ventilation_load():
    """Internal latent gains (W) enter the load additively. Doubling
    gains_internal_latent from 0 to X must add exactly X to p_lat[t]
    whenever the coil is on."""
    idx = _index()
    weather = _weather(idx, t_out_c=30.0, rh=0.70, p_pa=101325.0)
    common = dict(
        weather=weather,
        p_cool_sensible=np.full(24, 1.0),
        p_cool_max=np.full(24, np.inf),
        h_ve=np.full(24, 100.0),
        temp_max_c=24.0,
        target_humidity_rel=0.5,
    )
    _, p_lat_no_gain = compute_latent_cooling(gains_internal_latent=np.zeros(24), **common)
    gain = 250.0
    _, p_lat_with_gain = compute_latent_cooling(gains_internal_latent=np.full(24, gain), **common)

    assert np.all(np.isclose(p_lat_with_gain - p_lat_no_gain, gain, rtol=1e-6))


# --- Total-capacity capping (sensible priority) ------------------------------


def test_total_cap_undersized_prefers_sensible_and_clips_latent():
    """When the total cap is smaller than sensible + latent, sensible is
    served first (matches DX coil thermostat priority) and latent is
    clipped to the remainder. If the cap is even smaller than sensible
    alone, sensible is clipped and latent is zero."""
    idx = _index(n=4)
    weather = _weather(idx, t_out_c=30.0, rh=0.70, p_pa=101325.0)

    p_sens_in = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    caps = np.array(
        [
            np.inf,  # no cap → sensible + latent
            1500.0,  # cap between sensible and total → clip latent to 500
            800.0,  # cap below sensible → clip sensible to 800, latent to 0
            0.0,  # zero cap → both zero
        ]
    )

    p_sens_out, p_lat_out = compute_latent_cooling(
        weather=weather,
        p_cool_sensible=p_sens_in,
        p_cool_max=caps,
        h_ve=np.full(4, 100.0),
        gains_internal_latent=np.zeros(4),
        temp_max_c=24.0,
        target_humidity_rel=0.5,
    )

    # 1) uncapped: sensible untouched, latent > 0
    assert p_sens_out[0] == pytest.approx(1000.0)
    assert p_lat_out[0] > 0
    # 2) mid cap: sensible untouched, latent = cap − sensible
    assert p_sens_out[1] == pytest.approx(1000.0)
    assert p_sens_out[1] + p_lat_out[1] == pytest.approx(1500.0)
    # 3) tight cap: sensible clipped, latent forced to zero
    assert p_sens_out[2] == pytest.approx(800.0)
    assert p_lat_out[2] == 0.0
    # 4) zero cap: both zero
    assert p_sens_out[3] == 0.0
    assert p_lat_out[3] == 0.0


def test_returned_arrays_have_same_dtype_and_length_as_input():
    """The return contract: (p_sensible_out, p_latent) same shape/dtype as
    p_cool_sensible input. Downstream _format_output relies on this."""
    idx = _index()
    weather = _weather(idx, t_out_c=30.0, rh=0.7, p_pa=101325.0)
    p_sens = np.full(24, 500.0, dtype=np.float32)
    p_sens_out, p_lat = compute_latent_cooling(
        weather=weather,
        p_cool_sensible=p_sens,
        p_cool_max=np.full(24, np.inf, dtype=np.float32),
        h_ve=np.full(24, 100.0, dtype=np.float32),
        gains_internal_latent=np.zeros(24, dtype=np.float32),
        temp_max_c=24.0,
        target_humidity_rel=0.5,
    )
    assert p_sens_out.shape == p_sens.shape == p_lat.shape
    assert p_sens_out.dtype == p_lat.dtype
