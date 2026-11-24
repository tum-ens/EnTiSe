import logging

import numpy as np
import pandas as pd

from entise._optional import require_module
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Constants as Const
from entise.constants import Objects as O
from entise.core._time import RUN_TIME_KEY, output_index, run_time
from entise.core.base import Method
from entise.core.utils import resolve_ts_or_scalar
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.methods.auxiliary.ventilation.selector import Ventilation
from entise.methods.hvac._latent_cooling import compute_latent_cooling
from entise.methods.hvac._typed import TYPED, TypedViews
from entise.methods.hvac._weather import check_window_irradiance, hvac_weather, step_seconds
from entise.methods.hvac._windows import windows_table
from entise.methods.hvac.defaults import (
    DEFAULT_ACTIVE_COOLING,
    DEFAULT_ACTIVE_HEATING,
    DEFAULT_DEADBAND,
    DEFAULT_GAINS_INTERNAL_LATENT,
    DEFAULT_POWER_COOLING,
    DEFAULT_POWER_HEATING,
    DEFAULT_TARGET_HUMIDITY_REL,
    DEFAULT_TEMP_INIT,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_VENTILATION,
)

logger = logging.getLogger(__name__)


class R1C1(TypedViews, Method):
    """1R1C thermal RC model (single resistance, single capacitance).

    Purpose and scope
        Computes indoor air temperature and heating/cooling load required
        to keep setpoints using a minimal grey-box model with one thermal
        resistance (R) to ambient and one thermal capacitance (C)
        representing the zone's thermal inertia. Ventilation losses and
        internal/solar gains can be accounted for via auxiliary strategies
        or direct inputs.

    Model sketch (discrete time)
        ``T_in[t+1] = T_in[t] + (Δt/C) · ( (T_out[t] − T_in[t])/R
        + G_int[t] + G_sol[t] + H_ve[t]·(T_out[t] − T_in[t])
        + P_heat[t] − P_cool[t] )``. ``P_heat`` / ``P_cool`` are clipped to
        maximum device powers and only active when setpoint violations
        would occur.

    Typical use
        Quick load estimates, large batch simulations, or control studies
        where a lightweight model is sufficient and detailed envelope
        dynamics are not required. For more detailed dynamics, consider
        5R1C (ISO 13790) or 7R2C (VDI 6007).

    Typed heating and cooling
        Select the model for ``heating``, ``cooling`` or both, instead of
        ``hvac``, to get typed results from one solve. Each typed result has
        the indoor temperature and the columns and summary keys of its type.
        The control of a type the model is not selected for is off. A
        type-scoped control input, such as ``heating:power_heating[W]``,
        applies to its own control. ``hvac`` and a typed key cannot select
        the same model on one object.

    References
        Grey-box RC modeling of buildings (overview): many texts incl.
        ISO 13790 annexes; see also VDI 6007 for extended multi-node
        approaches.
    """

    types = [Types.HVAC, Types.HEATING, Types.COOLING]
    name = "1R1C"
    required_keys = [O.CAPACITANCE, O.RESISTANCE, O.WEATHER]
    optional_keys = {
        O.TIMEZONE: None,
        O.POWER_HEATING: DEFAULT_POWER_HEATING,
        O.POWER_COOLING: DEFAULT_POWER_COOLING,
        O.ACTIVE_HEATING: DEFAULT_ACTIVE_HEATING,
        O.ACTIVE_COOLING: DEFAULT_ACTIVE_COOLING,
        O.ACTIVE_GAINS_INTERNAL: True,
        O.ACTIVE_GAINS_SOLAR: True,
        O.ACTIVE_VENTILATION: True,
        O.TEMP_INIT: DEFAULT_TEMP_INIT,
        O.TEMP_MIN: DEFAULT_TEMP_MIN,
        O.TEMP_MAX: DEFAULT_TEMP_MAX,
        O.DEADBAND: DEFAULT_DEADBAND,
        # Latent cooling (issue #103)
        O.TARGET_HUMIDITY_REL: DEFAULT_TARGET_HUMIDITY_REL,
        O.GAINS_INTERNAL_LATENT: DEFAULT_GAINS_INTERNAL_LATENT,
        # Geometry
        O.AREA: Const.DEFAULT_AREA.value,
        O.HEIGHT: Const.DEFAULT_HEIGHT.value,
        # Location, needed for solar gains through windows
        O.LAT: None,
        O.LON: None,
        O.GAINS_INTERNAL: None,
        O.GAINS_INTERNAL_COL: None,
        O.VENTILATION: DEFAULT_VENTILATION,
        O.VENTILATION_COL: None,
        O.GAINS_SOLAR: None,
        O.WINDOWS: None,
    }
    required_data = [O.WEATHER]
    optional_data = [O.WINDOWS, O.GAINS_INTERNAL, O.GAINS_SOLAR, O.VENTILATION]
    output_summary = {
        f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": "total heating demand",
        f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": "maximum heating load",
        f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": "total cooling demand (sensible + latent)",
        f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": "maximum cooling load (sensible + latent)",
    }
    output_timeseries = {
        f"{C.TEMP_IN}": "indoor air temperature",
        f"{Types.HEATING}{SEP}{C.LOAD}[W]": "heating load",
        f"{Types.COOLING}{SEP}{C.LOAD}[W]": "total cooling load (sensible + latent)",
        f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]": "sensible cooling load",
        f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]": "latent cooling load",
    }

    input_aliases = {"temp_init": O.TEMP_INIT, "temp_min": O.TEMP_MIN, "temp_max": O.TEMP_MAX}

    def generate(
        self,
        obj: dict | None = None,
        data: dict | None = None,
        results: dict | None = None,
        ts_type: str = Types.HVAC,
        **inputs,
    ) -> dict:
        """Generate HVAC time series based on input parameters and weather data.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the indoor temperature and energy
        demand time series, and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing building parameters.
            data (dict, optional): Dictionary containing input data.
            results (dict, optional): Dictionary with results from previously generated time series
            ts_type (str, optional): Time series type to generate. Defaults to Types.HVAC.
            **inputs: Any input of the method, by its key or by its Python name (the key
                without its unit, such as ``area`` for ``area[m2]``). Keywords override the
                same keys in ``obj``. A table input, such as ``weather``, can be a
                DataFrame. For example:

                - ``capacitance`` (``capacitance[J K-1]``): Thermal capacitance in J/K.
                - ``resistance`` (``resistance[K W-1]``): Thermal resistance in K/W.
                - ``weather`` (DataFrame or str): Weather data with outdoor temperature.
                - ``power_heating`` and ``power_cooling`` (``[W]``, float or Series): Maximum heating
                  and cooling power in W.
                - ``active_heating`` and ``active_cooling`` (bool): Whether heating and cooling are active.
                - ``ventilation`` (``ventilation[W K-1]``): Ventilation rate in W/K.
                - ``temp_init``, ``temp_min`` and ``temp_max`` (``init_temperature[C]``,
                  ``min_temperature[C]``, ``max_temperature[C]``): Initial, minimum and maximum
                  indoor temperature in °C.
                - ``gains_internal`` (``gains_internal[W]``): Internal SENSIBLE heat gains in W.
                  Do NOT include the latent portion here — pass that via
                  ``gains_internal_latent``. ASHRAE metabolic tables split a seated occupant
                  as ~75 W sensible + ~55 W latent; the sensible number belongs here.
                - ``gains_solar`` (``gains_solar[W]``): Solar heat gains in W.
                - ``area`` (``area[m2]``) and ``height`` (``height[m]``): Heated area in m² and height in m.

        Returns:
            dict: Dictionary containing:
                - "summary" (dict): Summary statistics including total heating and
                  cooling demand, and maximum heating and cooling loads.
                - "timeseries" (pd.DataFrame): Time series of indoor temperature,
                  heating load, and cooling load with timestamps as index.

        Raises:
            Exception: If required data is missing or invalid.
        """
        obj, data = self._process_kwargs(obj, data, ts_type, **inputs)
        if ts_type in TYPED:
            return self.group_result(obj, data, results, ts_type)
        require_module("hvac-1r1c", "pvlib")
        # Read the tables once for this call; every helper below reuses this run.
        data = {**data, RUN_TIME_KEY: run_time(data)}
        obj, data = self._get_input_data(obj, data, ts_type)

        # Regular time step in seconds; each row is the interval [t, t + step).
        timestep = np.float32(step_seconds(data[O.WEATHER]))

        # Precompute auxiliary data
        data[O.GAINS_INTERNAL] = InternalGains().generate(obj, data)
        data[O.GAINS_SOLAR] = SolarGains().generate(obj, data)
        data[O.VENTILATION] = Ventilation().generate(obj, data)

        # Compute temperature and energy demand
        temp_in, p_heat, p_cool = calculate_timeseries_1r1c(obj, data, timestep)

        # Latent-cooling post-pass (issue #103). Decoupled from the T_in
        # dynamics — 1R1C carries no humidity state — so this runs vectorised
        # after the sensible solver, leaving the numba path untouched.
        # `p_cool` returned by the solver is already sensible-clipped to
        # `power_cooling` (the total cap); the post-pass may clip it further
        # if latent load pushes the total over that cap.
        weather_df = data[O.WEATHER]
        p_cool_max_arr = resolve_ts_or_scalar(obj, data, O.POWER_COOLING, weather_df.index).to_numpy(
            dtype=np.float32, copy=False
        )
        h_ve_arr = data[O.VENTILATION].to_numpy(dtype=np.float32, copy=False).ravel()
        gains_lat_arr = resolve_ts_or_scalar(obj, data, O.GAINS_INTERNAL_LATENT, weather_df.index).to_numpy(
            dtype=np.float32, copy=False
        )
        active_cool = bool(obj[O.ACTIVE_COOLING])
        if active_cool:
            p_cool_sensible, p_cool_latent = compute_latent_cooling(
                weather=weather_df,
                p_cool_sensible=p_cool,
                p_cool_max=p_cool_max_arr,
                h_ve=h_ve_arr,
                gains_internal_latent=gains_lat_arr,
                temp_max_c=float(obj[O.TEMP_MAX]),
                target_humidity_rel=float(obj[O.TARGET_HUMIDITY_REL]),
            )
        else:
            p_cool_sensible = p_cool
            p_cool_latent = np.zeros_like(p_cool)

        logger.debug(
            f"[HVAC R1C1] {ts_type}: max heating {p_heat.max()}, "
            f"cooling sensible {p_cool_sensible.max()}, latent {p_cool_latent.max()}"
        )

        return self._format_output(temp_in, p_heat, p_cool_sensible, p_cool_latent, obj, data, timestep)

    @staticmethod
    def _get_input_data(obj: dict, data: dict, method_type: str = Types.HVAC) -> tuple[dict, dict]:
        """Process and validate input data for HVAC calculation.

        This function extracts required and optional parameters from the input dictionaries,
        applies default values where needed, performs data validation, and prepares the
        data for HVAC calculation.

        Args:
            obj (dict): Dictionary containing building parameters such as thermal properties
                and temperature setpoints.
            data (dict): Dictionary containing input data such as weather information.
            method_type (str, optional): Method type to use for prefixing. Defaults to Types.HVAC.

        Returns:
            tuple: A tuple containing:
                - obj_out (dict): Processed object parameters with defaults applied.
                - data_out (dict): Processed data with required format for calculation.

        Notes:
            - Parameters can be specified with method-specific prefixes (e.g., "hvac:temp_min")
              which will take precedence over generic parameters (e.g., "temp_min").
        """
        obj_out = R1C1.get_relevant_objects(obj, method_type)
        run, weather_key, weather = hvac_weather(obj, data, method_type)
        windows = windows_table(obj, data, method_type)
        check_window_irradiance(run, weather_key, weather, obj, windows, method_type)
        internal_key = Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type)
        internal_gains = Method.get_with_backup(data, internal_key) if isinstance(internal_key, str) else None
        ventilation_key = Method.get_with_method_backup(obj, O.VENTILATION, method_type)
        ventilation = Method.get_with_backup(data, ventilation_key) if isinstance(ventilation_key, str) else None
        data_out = {
            RUN_TIME_KEY: run,
            O.WEATHER: weather,
            O.WINDOWS: windows,
            internal_key: internal_gains,
            ventilation_key: ventilation,
        }

        # Clean up
        data_out = {k: v for k, v in data_out.items() if v is not None}

        return obj_out, data_out

    @staticmethod
    def _format_output(temp_in, p_heat, p_cool_sensible, p_cool_latent, obj, data, timestep) -> dict:
        # NB: use ndarray.max() (C-level, single pass over the buffer), not
        # builtins.max() which iterates 8760 elements as Python objects.
        # `cooling:load[W]` and its summary counterparts carry the TOTAL
        # cooling load (sensible + latent). When latent is zero (no humidity
        # data, active_cooling=False, or ω_out ≤ ω_target with no internal
        # latent gain), total == sensible and pre-latent numbers are
        # preserved bit-for-bit.
        # Round sensible and latent first, then derive total as their sum so
        # the per-row invariant `total == sensible + latent` holds for
        # downstream consumers that assert equality — matches how R5C1/R7C2
        # emit the same columns.
        p_cool_sensible_int = p_cool_sensible.round().astype(int)
        p_cool_latent_int = p_cool_latent.round().astype(int)
        p_cool_total_int = p_cool_sensible_int + p_cool_latent_int
        # Demand still uses the un-rounded sum to preserve energy accuracy;
        # load_max uses the rounded-and-summed total so the summary number
        # is consistent with the DataFrame max.
        p_cool_total_float = p_cool_sensible + p_cool_latent
        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": int(round(p_heat.sum() * timestep / 3600)),
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": int(p_heat.max()),
            f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": int(round(p_cool_total_float.sum() * timestep / 3600)),
            f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": int(p_cool_total_int.max()),
        }

        df = pd.DataFrame(
            {
                f"{C.TEMP_IN}": temp_in.round(3),
                f"{Types.HEATING}{SEP}{C.LOAD}[W]": p_heat.round().astype(int),
                f"{Types.COOLING}{SEP}{C.LOAD}[W]": p_cool_total_int,
                f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]": p_cool_sensible_int,
                f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]": p_cool_latent_int,
            },
            index=output_index(data[O.WEATHER].index, obj, data),
        )

        return {"summary": summary, "timeseries": df}


# Below the small-G_tot threshold the analytical form (1 − exp(−x))/G_tot
# becomes numerically 0/0. Envelope conductances for real buildings are
# ≥ ~1 W/K, so this branch is defensive only.
_G_TOT_EPS = np.float32(1e-9)


def calculate_timeseries_1r1c(
    obj: dict, data: dict, timestep: float | np.floating
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to numpy or numba path based on the active accelerator.

    The two paths implement the same physics (analytical exponential
    update) and produce numerically identical results up to a few ULPs.
    See ``_calculate_timeseries_numpy`` for the physics and the private
    ``entise.methods.hvac._R1C1_numba`` module for the JIT-compiled
    variant.

    The accelerator is chosen by ``entise.get_accelerator``, controlled
    by the ``ENTISE_ACCELERATOR`` environment variable and the
    ``entise.set_accelerator`` runtime API.
    """
    from entise.perf import get_accelerator

    if get_accelerator() == "numba":
        # Lazy import: numba is optional (`pip install entise[numba]`).
        from entise.methods.hvac._R1C1_numba import calculate_timeseries_1r1c as _numba

        return _numba(obj, data, timestep)
    return _calculate_timeseries_numpy(obj, data, timestep)


def _calculate_timeseries_numpy(
    obj: dict, data: dict, timestep: float | np.floating
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate HVAC time series using a 1R1C model (pure numpy path).

    Integrates the lumped-capacitance ODE

        C · dT/dt = G_tot · (T_ss − T) + P_h − P_c ,
        G_tot = 1/R + H_ve ,
        T_ss  = T_out + (G_int + G_sol) / G_tot ,

    with the **analytical exponential update** for each step under piecewise-constant
    forcings. This solution is exact (up to float32 round-off) for constant
    T_out/H_ve/gains over one step and is unconditionally stable for any Δt/τ.

    Per-step impulse response (decay, gain) and passive steady state (T_ss_pas)
    depend only on the forcings — not on the previous temperature — so they are
    precomputed once as numpy arrays. Only the temperature recursion and the
    controller inversion remain inside the Python loop.

    Args:
        obj (dict): Building parameters (R, C, setpoints, power limits, flags).
        data (dict): Time-series inputs (weather, gains, ventilation conductance).
        timestep (float): Simulation step in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (indoor_temperature, heating_power, cooling_power).
    """
    # Scalar parameters
    thermal_capacitance = np.float32(obj[O.CAPACITANCE])
    temp_init = np.float32(obj[O.TEMP_INIT])
    temp_min = np.float32(obj[O.TEMP_MIN])
    temp_max = np.float32(obj[O.TEMP_MAX])
    active_heat = bool(obj[O.ACTIVE_HEATING])
    active_cool = bool(obj[O.ACTIVE_COOLING])
    inv_resistance = np.float32(1.0) / np.float32(obj[O.RESISTANCE])
    dt = np.float32(timestep)
    # Symmetric thermostat dead band. deadband = 0 collapses to the
    # aim-for-setpoint behavior (bit-exact regression).
    deadband = np.float32(obj.get(O.DEADBAND, DEFAULT_DEADBAND))
    temp_min_hi = temp_min + deadband  # upper band edge for heating
    temp_max_lo = temp_max - deadband  # lower band edge for cooling

    # Time-series inputs
    weather = data[O.WEATHER]
    temp_air = weather[C.TEMP_AIR].to_numpy(dtype=np.float32, copy=False)
    solar_gains = data[O.GAINS_SOLAR].to_numpy(dtype=np.float32, copy=False).ravel()
    internal_gains = data[O.GAINS_INTERNAL].to_numpy(dtype=np.float32, copy=False).ravel()
    ventilation = data[O.VENTILATION].to_numpy(dtype=np.float32, copy=False).ravel()

    # Power
    power_heat_max_arr = resolve_ts_or_scalar(
        obj, data, O.POWER_HEATING, weather.index, default=DEFAULT_POWER_HEATING
    ).to_numpy(dtype=np.float32, copy=False)
    power_cool_max_arr = resolve_ts_or_scalar(
        obj, data, O.POWER_COOLING, weather.index, default=DEFAULT_POWER_COOLING
    ).to_numpy(dtype=np.float32, copy=False)
    n_steps = temp_air.shape[0]

    # Vectorized precompute of the impulse response and passive steady state.
    # `g_tot_safe` guards against divide-by-zero in the pathological
    # small-G_tot branch; np.where then selects the correct value.
    g_tot = inv_resistance + ventilation
    safe = g_tot > _G_TOT_EPS
    g_tot_safe = np.where(safe, g_tot, np.float32(1.0))
    one_minus_decay = (-np.expm1(-(dt * g_tot_safe / thermal_capacitance))).astype(np.float32)
    dt_over_cap = dt / thermal_capacitance
    decay = np.where(safe, np.float32(1.0) - one_minus_decay, np.float32(1.0)).astype(np.float32)
    gain = np.where(safe, one_minus_decay / g_tot_safe, dt_over_cap).astype(np.float32)
    t_ss_pas = np.where(safe, temp_air + (solar_gains + internal_gains) / g_tot_safe, temp_air).astype(np.float32)

    temp_in = np.empty(n_steps, dtype=np.float32)
    p_heat = np.zeros(n_steps, dtype=np.float32)
    p_cool = np.zeros(n_steps, dtype=np.float32)
    temp_in[0] = temp_init

    # Scalar recursion — only the T_prev-dependent work stays in Python.
    #
    # For each step:
    #   T_pas_next = T_ss_pas + (T_prev - T_ss_pas) * decay        [no HVAC]
    #   T_next     = T_pas_next + gain * (P_h - P_c)                [with HVAC]
    #
    # Hysteresis state machine (see issue #102). `heating_on` / `cooling_on`
    # track whether the actuator was firing in the previous step. Heating
    # fires when either (a) it was off and T dropped below T_min, or (b) it
    # was on and T is still under the upper band edge (T_min + deadband).
    # While firing, the controller aims to land T_next on the upper band
    # edge (which collapses to T_min when deadband = 0). Cooling mirrors.
    temp_prev = temp_in[0]
    heating_on = False
    cooling_on = False
    # Row t is the interval [t, t + step): it starts at temp_in[t], is driven by
    # the forcing of row t and ends at temp_in[t + 1].
    for t in range(n_steps):
        d = decay[t]
        g = gain[t]
        t_ss = t_ss_pas[t]
        t_pas = t_ss + (temp_prev - t_ss) * d

        # A wide band (deadband >= (T_max - T_min)/2) can leave both latches
        # armed inside overlapping ranges. Clear the opposite latch whenever
        # the primary extreme trigger fires — physically we cannot heat while
        # already above T_max, nor cool while below T_min. With deadband = 0
        # these guards fire only in the degenerate case T_min == T_max and are
        # otherwise no-ops (bit-exact regression preserved).
        if t_pas > temp_max:
            heating_on = False
        if t_pas < temp_min:
            cooling_on = False

        p_h = np.float32(0.0)
        if active_heat:
            fire_h = t_pas < temp_min or (heating_on and t_pas < temp_min_hi)
            if fire_h:
                p_h_cap = power_heat_max_arr[t]
                need = (temp_min_hi - t_pas) / g
                p_h = need if need < p_h_cap else p_h_cap
                heating_on = True
            else:
                heating_on = False
        else:
            heating_on = False

        p_c = np.float32(0.0)
        if active_cool:
            fire_c = t_pas > temp_max or (cooling_on and t_pas > temp_max_lo)
            if fire_c:
                p_c_cap = power_cool_max_arr[t]
                need = (t_pas - temp_max_lo) / g
                p_c = need if need < p_c_cap else p_c_cap
                cooling_on = True
            else:
                cooling_on = False
        else:
            cooling_on = False

        p_heat[t] = p_h
        p_cool[t] = p_c
        temp_prev = t_pas + g * (p_h - p_c)
        if t + 1 < n_steps:
            temp_in[t + 1] = temp_prev

    return temp_in, p_heat, p_cool
