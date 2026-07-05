"""Numba-accelerated 1R1C solver.

Private module. Imported lazily from ``entise.methods.hvac.R1C1`` when the
active accelerator is ``'numba'`` (see :mod:`entise.perf`). Do not import
directly — call :func:`entise.methods.hvac.R1C1.calculate_timeseries_1r1c`
and let the dispatcher choose the path.

Design notes
------------

The numpy path in ``R1C1.py`` vectorizes the impulse-response precompute
(``G_tot``, ``decay``, ``gain``, ``T_ss_pas``) before the scalar recursion.
Under numba that pre-computation is counterproductive: allocating four
float32 arrays and reading from them at each iteration costs more than
letting the JIT keep the same scalars in registers and recompute them
per step. See ``bench_optim.py`` prototype comparison — variant V3
(numba over the per-step-recompute loop) beats V4 (numba + vectorized
precompute) on every case tested.

So this module recomputes ``G_tot``, ``decay``, ``gain``, ``T_ss_pas``
inside the ``@njit`` loop. All helpers are inlined for the same reason.

``fastmath`` is left off to preserve bit-for-bit reproducibility with the
numpy path — the numba win over numpy is already 100–300×, so the extra
10–20% from fastmath is not worth the numerical drift.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from entise.constants import Columns as C
from entise.constants import Objects as O

# Matches the constant in R1C1.py — kept in sync manually since the numba
# module cannot import from R1C1 without a cycle.
_G_TOT_EPS = np.float32(1e-9)


@njit(cache=True)
def _solve(
    T_out: np.ndarray,
    G_sol: np.ndarray,
    G_int: np.ndarray,
    H_ve: np.ndarray,
    inv_R: np.float32,
    dt: np.float32,
    C_th: np.float32,
    temp_init: np.float32,
    temp_min: np.float32,
    temp_max: np.float32,
    active_heat: bool,
    active_cool: bool,
    P_h_max: np.float32,
    P_c_max: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical exponential update, per-step recompute, JIT-compiled.

    Physics identical to the numpy path in ``R1C1.calculate_timeseries_1r1c``.
    """
    n = T_out.shape[0]
    temp_in = np.empty(n, dtype=np.float32)
    p_heat = np.zeros(n, dtype=np.float32)
    p_cool = np.zeros(n, dtype=np.float32)
    temp_in[0] = temp_init
    temp_prev = temp_in[0]
    dt_over_cap = dt / C_th

    for t in range(1, n):
        g_tot = inv_R + H_ve[t]
        if g_tot > _G_TOT_EPS:
            x = dt * g_tot / C_th
            one_minus_decay = -np.expm1(-x)
            decay = np.float32(1.0) - one_minus_decay
            gain = one_minus_decay / g_tot
            t_ss = T_out[t] + (G_sol[t] + G_int[t]) / g_tot
        else:
            decay = np.float32(1.0)
            gain = dt_over_cap
            t_ss = T_out[t]

        t_pas = t_ss + (temp_prev - t_ss) * decay

        p_h = np.float32(0.0)
        if active_heat and t_pas < temp_min:
            need = (temp_min - t_pas) / gain
            p_h = need if need < P_h_max else P_h_max

        p_c = np.float32(0.0)
        if active_cool and t_pas > temp_max:
            need = (t_pas - temp_max) / gain
            p_c = need if need < P_c_max else P_c_max

        p_heat[t] = p_h
        p_cool[t] = p_c
        temp_prev = t_pas + gain * (p_h - p_c)
        temp_in[t] = temp_prev
    return temp_in, p_heat, p_cool


def calculate_timeseries_1r1c(obj: dict, data: dict, timestep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba entry point matching the signature of the numpy dispatch target.

    Unpacks the obj/data dicts into flat arrays and scalars, then delegates
    to the ``@njit`` kernel above.
    """
    weather = data[O.WEATHER]
    T_out = weather[C.TEMP_AIR].to_numpy(dtype=np.float32, copy=False)
    G_sol = data[O.GAINS_SOLAR].to_numpy(dtype=np.float32, copy=False).ravel()
    G_int = data[O.GAINS_INTERNAL].to_numpy(dtype=np.float32, copy=False).ravel()
    H_ve = data[O.VENTILATION].to_numpy(dtype=np.float32, copy=False).ravel()

    return _solve(
        T_out,
        G_sol,
        G_int,
        H_ve,
        np.float32(1.0) / np.float32(obj[O.RESISTANCE]),
        np.float32(timestep),
        np.float32(obj[O.CAPACITANCE]),
        np.float32(obj[O.TEMP_INIT]),
        np.float32(obj[O.TEMP_MIN]),
        np.float32(obj[O.TEMP_MAX]),
        bool(obj[O.ACTIVE_HEATING]),
        bool(obj[O.ACTIVE_COOLING]),
        np.float32(obj[O.POWER_HEATING]),
        np.float32(obj[O.POWER_COOLING]),
    )
