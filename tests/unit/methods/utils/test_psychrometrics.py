"""Unit tests for the psychrometric helpers used by the HVAC latent-cooling
post-pass (issue #103).

Reference values are cross-checked against the ASHRAE Handbook — Fundamentals
Ch. 1 psychrometric chart (2017 edition). Tolerances are chosen tight enough
to reject a broken formula but loose enough to allow small differences between
Magnus-family saturation-pressure formulations.
"""

import numpy as np
import pytest

from entise.methods.utils.psychrometrics import (
    LATENT_HEAT_VAPORISATION,
    humidity_ratio,
    saturation_pressure_pa,
)

# --- Saturation pressure -----------------------------------------------------


@pytest.mark.parametrize(
    "temp_c, expected_pa, tol_pa",
    [
        # ASHRAE Ch. 1, Table 3 (2017) reference values
        (0.0, 611.2, 5.0),
        (10.0, 1228.0, 10.0),
        (20.0, 2338.8, 20.0),
        (25.0, 3169.0, 30.0),
        (30.0, 4245.5, 40.0),
    ],
)
def test_saturation_pressure_matches_ashrae_table(temp_c, expected_pa, tol_pa):
    """Saturation vapour pressure over water must match ASHRAE tabulated
    values within the tolerance of the Magnus-family approximation."""
    assert saturation_pressure_pa(temp_c) == pytest.approx(expected_pa, abs=tol_pa)


def test_saturation_pressure_vectorised():
    """The helper must accept a numpy array and return an array of the same
    shape — vectorisation is a hard requirement for the post-pass."""
    temps = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    result = saturation_pressure_pa(temps)
    assert result.shape == temps.shape
    assert np.all(np.diff(result) > 0), "e_sat must be strictly increasing in T"


def test_saturation_pressure_monotonic_over_realistic_range():
    """Saturation pressure must be strictly monotonic on [-20, 50] °C —
    covers all Central-European ambient conditions."""
    temps = np.linspace(-20.0, 50.0, 200)
    result = saturation_pressure_pa(temps)
    assert np.all(np.diff(result) > 0)


# --- Humidity ratio ----------------------------------------------------------


def test_humidity_ratio_ashrae_reference_25c_50rh():
    """ω(RH=50%, T=25°C, p=101325 Pa) ≈ 0.00988 kg/kg — standard ASHRAE
    psychrometric-chart cross-check for the classic comfort-band midpoint."""
    result = humidity_ratio(rh=0.5, temp_c=25.0, p_pa=101325.0)
    assert result == pytest.approx(0.00988, abs=5e-5)


def test_humidity_ratio_ashrae_reference_20c_50rh():
    """ω(RH=50%, T=20°C, p=101325 Pa) ≈ 0.00727 kg/kg — verifies the formula
    at a lower dry-bulb where saturation pressure is much smaller."""
    result = humidity_ratio(rh=0.5, temp_c=20.0, p_pa=101325.0)
    assert result == pytest.approx(0.00727, abs=5e-5)


def test_humidity_ratio_zero_at_zero_rh():
    """RH = 0 → ω = 0 regardless of T and p. Trivial but a common corner."""
    assert humidity_ratio(rh=0.0, temp_c=25.0, p_pa=101325.0) == 0.0
    assert humidity_ratio(rh=0.0, temp_c=-10.0, p_pa=95000.0) == 0.0


def test_humidity_ratio_monotonic_in_rh():
    """At fixed T and p, ∂ω/∂RH > 0. Any implementation with a sign flip
    would break latent-load direction."""
    rhs = np.linspace(0.0, 1.0, 20)
    ws = np.array([humidity_ratio(rh=r, temp_c=25.0, p_pa=101325.0) for r in rhs])
    assert np.all(np.diff(ws) > 0)


def test_humidity_ratio_monotonic_in_temp():
    """At fixed RH and p, ω(T) is strictly increasing over 0-40 °C
    (saturation pressure grows superlinearly with T)."""
    temps = np.linspace(0.0, 40.0, 20)
    ws = np.array([humidity_ratio(rh=0.5, temp_c=t, p_pa=101325.0) for t in temps])
    assert np.all(np.diff(ws) > 0)


def test_humidity_ratio_vectorised_over_time():
    """All three inputs may be arrays; result must broadcast to a matching
    array. This is the shape actually consumed by the post-pass."""
    n = 24
    rh = np.full(n, 0.5)
    temps = np.linspace(15.0, 35.0, n)
    pressures = np.full(n, 101325.0)
    result = humidity_ratio(rh=rh, temp_c=temps, p_pa=pressures)
    assert result.shape == (n,)
    # Sanity: rising temperature at constant RH must raise ω.
    assert np.all(np.diff(result) > 0)


# --- Latent-heat constant ----------------------------------------------------


def test_latent_heat_of_vaporisation_constant():
    """The module exports a single latent-heat-of-vaporisation constant
    (~2.45 MJ/kg at typical coil conditions of ~15 °C). Value is the
    ASHRAE standard psychrometric-chart constant used across dehum
    calculations. Documented as an upper-bound approximation."""
    assert 2.4e6 < LATENT_HEAT_VAPORISATION < 2.5e6
