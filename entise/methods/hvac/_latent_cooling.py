"""Shared latent-cooling post-pass for the RC HVAC models (issue #103).

The 1R1C, 5R1C and 7R2C solvers are sensible-only: they track a single
dry-bulb air temperature and compute only the sensible portion of the
cooling load. Real cooling systems also remove humidity from the air —
in humid Central-European summers this is 30–50 % of the true cooling
demand.

Since none of the RC solvers carries a humidity state, the latent load is
decoupled from ``T_in`` dynamics: it depends only on the outdoor humidity
transported into the zone via ventilation, plus any latent internal gains.
It can therefore be computed as a pure post-pass over the sensible-load
result, without touching any solver kernel (including the numba path).

This module is private (leading underscore) and holds no ``Method`` subclass
so the auto-doc discovery skips it.

Convention for the two internal-gain inputs (worth reiterating because it's
an easy source of double-counting):

* ``gains_internal[W]`` is **sensible-only** — the heat-flux portion that
  raises air temperature.
* ``gains_internal_latent[W]`` is the **latent portion** — the moisture
  release that has to be dehumidified out.

ASHRAE metabolic tables split a seated occupant as ~75 W sensible + ~55 W
latent. Users migrating from a pre-latent EnTiSe simulation who lumped the
whole 130 W into ``gains_internal[W]`` should reduce it to the sensible
portion and move the latent portion into ``gains_internal_latent[W]``.

Semantics: see issue #103 for the full design. In brief:

* Latent load is only counted when sensible cooling is active
  (``p_cool_sensible > 0``). A standard split AC without a dehumidification
  mode does not dehumidify with the coil off.
* Bypass factor is 0 (perfect coil) — documented upper bound; a
  configurable BPF is a follow-up (issue #104).
* When the total cap ``p_cool_max`` is tighter than sensible + latent,
  sensible is served first and latent is clipped to the remainder. This
  matches thermostat priority on a DX coil.
* Missing ``relative_humidity[1]`` or ``surface_air_pressure[Pa]`` in the
  weather → the helper returns zeros for latent, emits one warning per
  missing column, and leaves sensible untouched.
* Negative ``gains_internal_latent`` is a valid input and models a
  moisture sink (e.g. a separate dehumidifier). It offsets the
  ventilation-driven load; if the sum goes negative the AC's latent
  contribution is clipped to zero — the coil can't run in reverse.

Known limitation — T_in consistency when the total cap binds
------------------------------------------------------------
The sensible solver clips its own output to ``p_cool_max`` and integrates
``T_in`` under that clip. When the cap binds and latent is non-zero, the
post-pass then re-attributes some of the total to latent — but ``T_in``
was already computed as if the entire cap were spent on sensible.
Consequence: a hypothetical coil-detail model that traded sensible for
latent under a total cap would predict a slightly higher ``T_in`` than
we report. Our "sensible-priority" behaviour matches how a
thermostat-controlled DX coil actually cycles, so the discrepancy shows
up only against coil-sizing tools that model the equipment in more
detail. Not fixable without an inner iteration on the solver — deferred.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from entise.constants import Columns as C
from entise.constants.constants import CP_AIR
from entise.methods.utils.psychrometrics import LATENT_HEAT_VAPORISATION, humidity_ratio

logger = logging.getLogger(__name__)


def compute_latent_cooling(
    weather: pd.DataFrame,
    p_cool_sensible: np.ndarray,
    p_cool_max: np.ndarray,
    h_ve: np.ndarray,
    gains_internal_latent: np.ndarray,
    temp_max_c,
    target_humidity_rel: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the latent cooling load and re-cap the sensible load.

    Args:
        weather: Weather DataFrame. Must carry ``C.TEMP_AIR``. Latent load
            requires ``C.HUMIDITY_REL`` and ``C.SURFACE_AIR_PRESSURE`` too;
            if either is missing the helper falls back to zero latent and
            emits a warning naming the missing columns.
        p_cool_sensible: Sensible cooling power per step (W). The output
            ``p_sensible_out`` matches this unless the total cap forces a
            clip.
        p_cool_max: Total cooling power cap per step (W). ``inf`` means no cap.
        h_ve: Ventilation heat-transfer coefficient per step (W/K). Used to
            back out the ventilation air mass flow as ``m_dot = H_ve / c_p_air``.
        gains_internal_latent: Latent internal gains per step (W). Added
            straight into the moisture load (already in load units, not
            moisture-flow units — matches the ``gains_internal[W]`` pattern).
        temp_max_c: Cooling setpoint used to evaluate the target humidity
            ratio. Scalar or array; broadcast against the per-step arrays.
        target_humidity_rel: Target indoor RH in [0, 1], evaluated at
            ``temp_max_c``. ASHRAE Standard 55 comfort-band midpoint is 0.5.

    Returns:
        (p_sensible_out, p_latent) — two numpy arrays with the same shape
        and dtype as ``p_cool_sensible``.
    """
    p_sens_in = np.asarray(p_cool_sensible)
    dtype = p_sens_in.dtype
    n = p_sens_in.shape[0]

    # Graceful degradation: any missing psychrometric input → zero latent,
    # sensible untouched, one warning per missing column.
    missing = [col for col in (C.HUMIDITY_REL, C.SURFACE_AIR_PRESSURE) if col not in weather.columns]
    if missing:
        logger.warning(
            "Latent cooling skipped — weather is missing column(s): %s. "
            "Add them to the weather DataFrame (OpenMeteo returns "
            "`relative_humidity_2m` and `surface_pressure` by default) to "
            "enable latent load.",
            ", ".join(missing),
        )
        return p_sens_in.astype(dtype, copy=True), np.zeros(n, dtype=dtype)

    # Per-step psychrometric evaluation. Kept in float64 for the humidity
    # arithmetic (small numbers relative to atmospheric pressure) and cast
    # back to the sensible-load dtype at the end.
    t_out = weather[C.TEMP_AIR].to_numpy(dtype=np.float64, copy=False)
    rh_out = weather[C.HUMIDITY_REL].to_numpy(dtype=np.float64, copy=False)
    p_pa = weather[C.SURFACE_AIR_PRESSURE].to_numpy(dtype=np.float64, copy=False)
    h_ve64 = np.asarray(h_ve, dtype=np.float64)
    gains_lat64 = np.asarray(gains_internal_latent, dtype=np.float64)

    omega_out = humidity_ratio(rh=rh_out, temp_c=t_out, p_pa=p_pa)
    omega_target = humidity_ratio(rh=target_humidity_rel, temp_c=temp_max_c, p_pa=p_pa)

    m_dot_air = h_ve64 / CP_AIR  # kg/s
    q_latent = m_dot_air * (omega_out - omega_target) * LATENT_HEAT_VAPORISATION + gains_lat64
    # AC can't add moisture — floor at zero.
    q_latent = np.maximum(q_latent, 0.0)

    # Latent only counts when the coil is on. Cast the mask against the
    # sensible input so downstream comparisons stay in the caller's dtype.
    coil_on = p_sens_in > 0
    q_latent = np.where(coil_on, q_latent, 0.0)

    # Sensible-priority capping: sensible is served first, latent gets the
    # remainder. ``inf`` cap flows through unchanged. When sensible alone
    # exceeds the cap, sensible is clipped down and latent goes to zero.
    p_cap64 = np.asarray(p_cool_max, dtype=np.float64)
    p_sens64 = p_sens_in.astype(np.float64, copy=False)
    p_sens_out = np.minimum(p_sens64, p_cap64)
    remaining = np.maximum(p_cap64 - p_sens_out, 0.0)
    p_latent_out = np.minimum(q_latent, remaining)

    return p_sens_out.astype(dtype, copy=False), p_latent_out.astype(dtype, copy=False)
