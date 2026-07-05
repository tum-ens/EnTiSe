"""Psychrometric helpers for humid-air calculations.

Pure numpy utility module — no ``Method`` subclass, kept out of the
auto-generated docs discovery on purpose. Reused by the latent-cooling
post-pass across the RC HVAC models (see :mod:`entise.methods.hvac._latent_cooling`).

Formulas follow the standard Magnus-Tetens approximation for saturation
pressure and the ideal-gas mixing-ratio definition for humidity ratio.
Reference: ASHRAE Handbook — Fundamentals, Ch. 1 (2017).
"""

from __future__ import annotations

import numpy as np

# --- Physical constants -----------------------------------------------------

# Latent heat of vaporisation of water at typical AC coil conditions
# (~15 °C dew point). Used as a single constant in the latent-cooling
# post-pass — good to ~2 % over the 0–35 °C range. Add a temperature-
# dependent form if a future issue calls for it.
LATENT_HEAT_VAPORISATION: float = 2.45e6  # J/kg

# Ratio of molar masses M_w / M_a (~ 18.015 / 28.965). Standard psychrometric
# constant used in the humidity-ratio definition.
_MW_OVER_MA: float = 0.622


def saturation_pressure_pa(temp_c):
    """Saturation vapour pressure of water over a flat liquid surface.

    Uses the Magnus (Tetens) formula::

        e_sat(T) = 611.2 · exp(17.62 · T / (243.12 + T))       [Pa, T in °C]

    Accurate to <0.5 % over -20 to 50 °C — the relevant range for ambient
    humidity in a building-simulation context.

    Args:
        temp_c: Temperature in °C. Scalar or numpy array.

    Returns:
        Saturation vapour pressure in Pa. Same shape as ``temp_c``.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    return 611.2 * np.exp(17.62 * t / (243.12 + t))


def humidity_ratio(rh, temp_c, p_pa):
    """Humidity ratio (mixing ratio) ω from RH, dry-bulb temperature and pressure.

    Standard definition::

        p_v = RH · e_sat(T)
        ω   = (M_w / M_a) · p_v / (p − p_v)                     [kg water / kg dry air]

    All arguments broadcast against each other; the result has the broadcast
    shape.

    Args:
        rh: Relative humidity as a fraction in [0, 1] (not %).
        temp_c: Dry-bulb temperature in °C.
        p_pa: Total (barometric) pressure in Pa.

    Returns:
        Humidity ratio ω in kg water / kg dry air.
    """
    rh_arr = np.asarray(rh, dtype=np.float64)
    p_v = rh_arr * saturation_pressure_pa(temp_c)
    p_arr = np.asarray(p_pa, dtype=np.float64)
    # Guard against p_v ≥ p (physically impossible, but a bad input could
    # cause a divide-by-zero here). Cap p_v strictly below p.
    p_v = np.minimum(p_v, p_arr * (1.0 - 1e-9))
    return _MW_OVER_MA * p_v / (p_arr - p_v)
