import logging

import numpy as np
import pandas as pd

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Constants as Const
from entise.constants import Objects as O
from entise.core.base import Method
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.methods.auxiliary.ventilation.selector import Ventilation
from entise.methods.hvac.defaults import (
    DEFAULT_ACTIVE_COOLING,
    DEFAULT_ACTIVE_HEATING,
    DEFAULT_POWER_COOLING,
    DEFAULT_POWER_HEATING,
    DEFAULT_TEMP_INIT,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_VENTILATION,
)

logger = logging.getLogger(__name__)

# Module-level caches (per process)
_WEATHER_CACHE: dict[tuple, pd.DataFrame] = {}


class R1C1(Method):
    """1R1C thermal RC model (single resistance, single capacitance).

    Purpose and scope:
      - Computes indoor air temperature and heating/cooling load required to keep
      setpoints using a minimal grey‑box model with one thermal resistance (R)
      to ambient and one thermal capacitance (C) representing the zone’s thermal
      inertia. Ventilation losses and internal/solar gains can be accounted for
      via auxiliary strategies or direct inputs.

    Model sketch (discrete time):
      - T_in[t+1] = T_in[t] + (Δt/C)·( (T_out[t] − T_in[t])/R + G_int[t] + G_sol[t] + H_ve[t]·(T_out[t] − T_in[t])
      + P_heat[t] − P_cool[t] )
      - P_heat/P_cool are clipped to maximum device powers and only active when
      setpoint violations would occur.

    Typical use:
    - Quick load estimates, large batch simulations, or control studies where a
      lightweight model is sufficient and detailed envelope dynamics are not
      required. For more detailed dynamics, consider 5R1C (ISO 13790) or 7R2C (VDI 6007).

    References:
    - Grey‑box RC modeling of buildings (overview): many texts incl. ISO 13790 annexes;
      see also VDI 6007 for extended multi‑node approaches.
    """

    types = [Types.HVAC]
    name = "1R1C"
    required_keys = [O.CAPACITANCE, O.RESISTANCE, O.WEATHER]
    optional_keys = [
        O.POWER_HEATING,
        O.POWER_COOLING,
        O.ACTIVE_HEATING,
        O.ACTIVE_COOLING,
        O.ACTIVE_GAINS_INTERNAL,
        O.ACTIVE_GAINS_SOLAR,
        O.ACTIVE_VENTILATION,
        O.TEMP_INIT,
        O.TEMP_MIN,
        O.TEMP_MAX,
        O.AREA,
        O.HEIGHT,
        O.GAINS_INTERNAL,
        O.GAINS_INTERNAL_COL,
        O.VENTILATION,
        O.VENTILATION_COL,
        O.GAINS_SOLAR,
        O.WINDOWS,
    ]
    required_data = [O.WEATHER]
    optional_data = [O.WINDOWS, O.GAINS_INTERNAL, O.GAINS_SOLAR, O.VENTILATION]
    output_summary = {
        f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": "total heating demand",
        f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": "maximum heating load",
        f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": "total cooling demand",
        f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": "maximum cooling load",
    }
    output_timeseries = {
        f"{C.TEMP_IN}": "indoor air temperature",
        f"{Types.HEATING}{SEP}{C.LOAD}[W]": "heating load",
        f"{Types.COOLING}{SEP}{C.LOAD}[W]": "cooling load",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        results: dict | None = None,
        ts_type: str = Types.HVAC,
        *,
        capacitance: float = None,
        resistance: float = None,
        weather: pd.DataFrame = None,
        power_heating: float = None,
        power_cooling: float = None,
        active_heating: bool = None,
        active_cooling: bool = None,
        ventilation: float = None,
        temp_init: float = None,
        temp_min: float = None,
        temp_max: float = None,
        gains_internal: pd.DataFrame = None,
        gains_solar: pd.DataFrame = None,
        area: float = None,
        height: float = None,
    ):
        """Generate HVAC time series based on input parameters and weather data.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the indoor temperature and energy
        demand time series, and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing building parameters.
            data (dict, optional): Dictionary containing input data.
            results (dict, optional): Dictionary with results from previously generated time series
            ts_type (str, optional): Time series type to generate. Defaults to Types.HVAC.
            capacitance (float, optional): Thermal capacitance in J/K.
            resistance (float, optional): Thermal resistance in K/W.
            weather (pd.DataFrame, optional): Weather data with outdoor temperature.
            power_heating (float, optional): Maximum heating power in W.
            power_cooling (float, optional): Maximum cooling power in W.
            active_heating (bool, optional): Whether heating is active.
            active_cooling (bool, optional): Whether cooling is active.
            ventilation (float, optional): Ventilation rate in W/K.
            temp_init (float, optional): Initial indoor temperature in °C.
            temp_min (float, optional): Minimum indoor temperature in °C.
            temp_max (float, optional): Maximum indoor temperature in °C.
            gains_internal (pd.DataFrame, optional): Internal heat gains in W.
            gains_solar (pd.DataFrame, optional): Solar heat gains in W.
            area (float, optional): Heated area in m².
            height (float, optional): Heated height in m³.

        Returns:
            dict: Dictionary containing:
                - "summary" (dict): Summary statistics including total heating and
                  cooling demand, and maximum heating and cooling loads.
                - "timeseries" (pd.DataFrame): Time series of indoor temperature,
                  heating load, and cooling load with timestamps as index.

        Raises:
            Exception: If required data is missing or invalid.
        """
        obj, data = self._process_kwargs(
            obj,
            data,
            capacitance=capacitance,
            resistance=resistance,
            weather=weather,
            power_heating=power_heating,
            power_cooling=power_cooling,
            active_heating=active_heating,
            active_cooling=active_cooling,
            ventilation=ventilation,
            temp_init=temp_init,
            temp_min=temp_min,
            temp_max=temp_max,
            gains_internal=gains_internal,
            gains_solar=gains_solar,
            area=area,
            height=height,
        )
        obj, data = self._get_input_data(obj, data, ts_type)

        # Timestep in seconds (assuming a regular time grid)
        idx = data[O.WEATHER][C.DATETIME].values.astype("datetime64[ns]")
        timestep = np.float32((idx[1] - idx[0]) / np.timedelta64(1, "s"))

        # Precompute auxiliary data
        data[O.GAINS_INTERNAL] = InternalGains().generate(obj, data)
        data[O.GAINS_SOLAR] = SolarGains().generate(obj, data)
        data[O.VENTILATION] = Ventilation().generate(obj, data)

        # Compute temperature and energy demand
        temp_in, p_heat, p_cool = calculate_timeseries_1r1c(obj, data, timestep)

        logger.debug(f"[HVAC R1C1] {ts_type}: max heating {p_heat.max()}, cooling {p_cool.max()}")

        return self._format_output(temp_in, p_heat, p_cool, data, timestep)

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
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            # Geometry
            O.AREA: Method.get_with_method_backup(obj, O.AREA, method_type, Const.DEFAULT_AREA.value),
            O.HEIGHT: Method.get_with_method_backup(obj, O.HEIGHT, method_type, Const.DEFAULT_HEIGHT.value),
            O.LAT: Method.get_with_method_backup(obj, O.LAT, method_type),
            O.LON: Method.get_with_method_backup(obj, O.LON, method_type),
            # Controls
            O.ACTIVE_COOLING: Method.get_with_method_backup(obj, O.ACTIVE_COOLING, method_type, DEFAULT_ACTIVE_COOLING),
            O.ACTIVE_HEATING: Method.get_with_method_backup(obj, O.ACTIVE_HEATING, method_type, DEFAULT_ACTIVE_HEATING),
            O.ACTIVE_GAINS_INTERNAL: Method.get_with_method_backup(obj, O.ACTIVE_GAINS_INTERNAL, method_type, True),
            O.ACTIVE_GAINS_SOLAR: Method.get_with_method_backup(obj, O.ACTIVE_GAINS_SOLAR, method_type, True),
            O.ACTIVE_VENTILATION: Method.get_with_method_backup(obj, O.ACTIVE_VENTILATION, method_type, True),
            # Gains
            O.GAINS_INTERNAL: Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type),
            O.GAINS_INTERNAL_COL: Method.get_with_method_backup(obj, O.GAINS_INTERNAL_COL, method_type),
            O.GAINS_SOLAR: Method.get_with_method_backup(obj, O.GAINS_SOLAR, method_type),
            # Power limits
            O.POWER_COOLING: Method.get_with_method_backup(obj, O.POWER_COOLING, method_type, DEFAULT_POWER_COOLING),
            O.POWER_HEATING: Method.get_with_method_backup(obj, O.POWER_HEATING, method_type, DEFAULT_POWER_HEATING),
            # 1R1C RC parameters
            O.RESISTANCE: Method.get_with_method_backup(obj, O.RESISTANCE, method_type),
            O.CAPACITANCE: Method.get_with_method_backup(obj, O.CAPACITANCE, method_type),
            # Temperature setpoints
            O.TEMP_INIT: Method.get_with_method_backup(obj, O.TEMP_INIT, method_type, DEFAULT_TEMP_INIT),
            O.TEMP_MAX: Method.get_with_method_backup(obj, O.TEMP_MAX, method_type, DEFAULT_TEMP_MAX),
            O.TEMP_MIN: Method.get_with_method_backup(obj, O.TEMP_MIN, method_type, DEFAULT_TEMP_MIN),
            # Ventilation
            O.VENTILATION: Method.get_with_method_backup(obj, O.VENTILATION, method_type, DEFAULT_VENTILATION),
            O.VENTILATION_COL: Method.get_with_method_backup(obj, O.VENTILATION_COL, method_type),
        }
        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather = Method.get_with_backup(data, weather_key)
        windows_key = Method.get_with_method_backup(obj, O.WINDOWS, method_type)
        windows = Method.get_with_backup(data, windows_key) if isinstance(windows_key, str) else None
        internal_key = Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type)
        internal_gains = Method.get_with_backup(data, internal_key) if isinstance(internal_key, str) else None
        ventilation_key = Method.get_with_method_backup(obj, O.VENTILATION, method_type)
        ventilation = Method.get_with_backup(data, ventilation_key) if isinstance(ventilation_key, str) else None
        data_out = {
            O.WEATHER: weather,
            O.WINDOWS: windows,
            internal_key: internal_gains,
            ventilation_key: ventilation,
        }

        # Clean up
        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        data_out = {k: v for k, v in data_out.items() if v is not None}

        # Safe datetime handling
        weather_cache_key = weather_key
        weather_cached = _WEATHER_CACHE.get(weather_cache_key)
        if weather_cached is None:
            if O.WEATHER in data_out:
                weather = data_out[O.WEATHER].copy()
                weather = Method._strip_weather_height(weather)
                weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME])
                weather.set_index(C.DATETIME, inplace=True, drop=False)
                data_out[O.WEATHER] = weather
                _WEATHER_CACHE[weather_cache_key] = weather
        else:
            data_out[O.WEATHER] = weather_cached

        return obj_out, data_out

    # @staticmethod
    # def _get_input_data(obj: dict, data: dict, method_type: str = Types.HVAC) -> tuple[dict, dict]:
    #     """Process and validate input data for HVAC calculation (R1C1).
    #
    #     - Resolves method-prefixed keys with fallback to shared keys
    #     - Normalizes and caches weather (index = Columns.DATETIME)
    #     - Passes through references to optional timeseries (windows, gains, ventilation)
    #     - Adds optional `O.H_VE` so users can pass ventilation conductance directly
    #     """
    #     obj_out = {
    #         O.ID: Method.get_with_backup(obj, O.ID),
    #         # Controls
    #         O.ACTIVE_COOLING: Method.get_with_method_backup(obj, O.ACTIVE_COOLING, method_type,
    #         DEFAULT_ACTIVE_COOLING),
    #         O.ACTIVE_HEATING: Method.get_with_method_backup(obj, O.ACTIVE_HEATING, method_type,
    #         DEFAULT_ACTIVE_HEATING),
    #         O.AREA: Method.get_with_method_backup(obj, O.AREA, method_type, Const.DEFAULT_AREA.value),
    #         O.GAINS_INTERNAL: Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type),
    #         O.GAINS_INTERNAL_COL: Method.get_with_method_backup(obj, O.GAINS_INTERNAL_COL, method_type),
    #         O.GAINS_SOLAR: Method.get_with_method_backup(obj, O.GAINS_SOLAR, method_type),
    #         O.HEIGHT: Method.get_with_method_backup(obj, O.HEIGHT, method_type, Const.DEFAULT_HEIGHT.value),
    #         O.LAT: Method.get_with_method_backup(obj, O.LAT, method_type),
    #         O.LON: Method.get_with_method_backup(obj, O.LON, method_type),
    #         O.POWER_COOLING: Method.get_with_method_backup(obj, O.POWER_COOLING, method_type, DEFAULT_POWER_COOLING),
    #         O.POWER_HEATING: Method.get_with_method_backup(obj, O.POWER_HEATING, method_type, DEFAULT_POWER_HEATING),
    #         O.RESISTANCE: Method.get_with_method_backup(obj, O.RESISTANCE, method_type),
    #         O.CAPACITANCE: Method.get_with_method_backup(obj, O.CAPACITANCE, method_type),
    #         O.TEMP_INIT: Method.get_with_method_backup(obj, O.TEMP_INIT, method_type, DEFAULT_TEMP_INIT),
    #         O.TEMP_MAX: Method.get_with_method_backup(obj, O.TEMP_MAX, method_type, DEFAULT_TEMP_MAX),
    #         O.TEMP_MIN: Method.get_with_method_backup(obj, O.TEMP_MIN, method_type, DEFAULT_TEMP_MIN),
    #         # Ventilation: allow both the auxiliary path and direct H_ve (conductance)
    #         O.VENTILATION: Method.get_with_method_backup(obj, O.VENTILATION, method_type, DEFAULT_VENTILATION),
    #         O.VENTILATION_COL: Method.get_with_method_backup(obj, O.VENTILATION_COL, method_type),
    #         O.H_VE: Method.get_with_method_backup(obj, O.H_VE, method_type),  # optional, may be scalar or Series
    #     }
    #
    #     weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
    #     weather = Method.get_with_backup(data, weather_key)
    #
    #     windows_key = Method.get_with_method_backup(obj, O.WINDOWS, method_type)
    #     windows = Method.get_with_backup(data, windows_key) if isinstance(windows_key, str) else None
    #
    #     internal_key = Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type)
    #     internal_gains = Method.get_with_backup(data, internal_key) if isinstance(internal_key, str) else None
    #
    #     ventilation_key = Method.get_with_method_backup(obj, O.VENTILATION, method_type)
    #     ventilation = Method.get_with_backup(data, ventilation_key) if isinstance(ventilation_key, str) else None
    #
    #     data_out = {
    #         O.WEATHER: weather,
    #         O.WINDOWS: windows,
    #         internal_key: internal_gains,
    #         ventilation_key: ventilation,
    #     }
    #
    #     # Clean up Nones
    #     obj_out = {k: v for k, v in obj_out.items() if v is not None}
    #     data_out = {k: v for k, v in data_out.items() if v is not None}
    #
    #     # Weather normalization and caching
    #     weather_cache_key = weather_key
    #     weather_cached = _WEATHER_CACHE.get(weather_cache_key)
    #     if weather_cached is None:
    #         if O.WEATHER in data_out:
    #             weather = data_out[O.WEATHER].copy()
    #             weather = Method._strip_weather_height(weather)
    #             weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME])
    #             weather.set_index(C.DATETIME, inplace=True, drop=False)
    #             data_out[O.WEATHER] = weather
    #             _WEATHER_CACHE[weather_cache_key] = weather
    #     else:
    #         data_out[O.WEATHER] = weather_cached
    #
    #     return obj_out, data_out

    def _prepare_inputs(self, obj: dict, data: dict) -> dict:
        """Prepare a solver-ready bundle for R1C1.

        Returns a dict with keys:
          - index: pd.DatetimeIndex
          - dt_s: float (seconds)
          - weather: pd.DataFrame
          - g_int_series: pd.Series [W]
          - g_sol_series: pd.Series [W]
          - Hve_series: pd.Series [W/K] (prefers O.H_VE if provided)
          - controls: dict (setpoints, caps, activation flags)
          - params: dict with R1C1 parameters (C, R)
        """
        # Weather and timestep
        weather = data[O.WEATHER]
        index = weather.index
        idx = weather[C.DATETIME].values.astype("datetime64[ns]")
        dt_s = float((idx[1] - idx[0]) / np.timedelta64(1, "s"))

        # Gains (R1C1 currently computes auxiliaries unconditionally, keep behavior)
        g_sol_df = data.get(O.GAINS_SOLAR) or SolarGains().generate(obj, {**data, O.WEATHER: weather})
        g_int_df = data.get(O.GAINS_INTERNAL) or InternalGains().generate(obj, {**data, O.WEATHER: weather})

        g_int_series = g_int_df.sum(axis=1) if isinstance(g_int_df, pd.DataFrame) else pd.Series(0.0, index=index)
        g_sol_series = g_sol_df.sum(axis=1) if isinstance(g_sol_df, pd.DataFrame) else pd.Series(0.0, index=index)

        # Ventilation normalization: prefer direct H_ve if present; otherwise use auxiliary/data/default
        hve_series: pd.Series
        if O.H_VE in obj and obj[O.H_VE] is not None:
            val = obj[O.H_VE]
            if isinstance(val, pd.Series):
                if not val.index.equals(index):
                    raise ValueError("H_ve series index does not match weather index.")
                hve_series = val.astype(float)
            else:
                try:
                    fval = float(val)
                    hve_series = pd.Series(np.full(len(index), fval, dtype=float), index=index, name=O.VENTILATION)
                except Exception as err:
                    raise ValueError(f"H_ve must be a float or a pandas Series, got {type(val)}") from err
        else:
            ven_df = data.get(O.VENTILATION)
            if isinstance(ven_df, pd.DataFrame) and O.VENTILATION in ven_df:
                hve_series = ven_df[O.VENTILATION].astype(float)
            elif O.VENTILATION in obj:
                try:
                    fval = float(obj[O.VENTILATION])
                    hve_series = pd.Series(np.full(len(index), fval, dtype=float), index=index, name=O.VENTILATION)
                except Exception:
                    ven_df = Ventilation().generate(obj, {**data, O.WEATHER: weather})
                    hve_series = ven_df[O.VENTILATION].astype(float)
            else:
                ven_df = Ventilation().generate(obj, {**data, O.WEATHER: weather})
                hve_series = ven_df[O.VENTILATION].astype(float)

        # Controls and parameters
        controls = dict(
            T_init=float(obj.get(O.TEMP_INIT, DEFAULT_TEMP_INIT)),
            T_min=float(obj.get(O.TEMP_MIN, DEFAULT_TEMP_MIN)),
            T_max=float(obj.get(O.TEMP_MAX, DEFAULT_TEMP_MAX)),
            P_h_max=float(obj.get(O.POWER_HEATING, DEFAULT_POWER_HEATING)),
            P_c_max=float(obj.get(O.POWER_COOLING, DEFAULT_POWER_COOLING)),
            on_h=bool(obj.get(O.ACTIVE_HEATING, DEFAULT_ACTIVE_HEATING)),
            on_c=bool(obj.get(O.ACTIVE_COOLING, DEFAULT_ACTIVE_COOLING)),
        )

        params = dict(
            C=float(obj[O.CAPACITANCE]),
            R=float(obj[O.RESISTANCE]),
        )

        return dict(
            index=index,
            dt_s=dt_s,
            weather=weather,
            g_int_series=g_int_series,
            g_sol_series=g_sol_series,
            Hve_series=hve_series,
            controls=controls,
            params=params,
        )

    @staticmethod
    def _format_output(temp_in, p_heat, p_cool, data, timestep) -> dict:
        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": int(round(p_heat.sum() * timestep / 3600)),
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": int(max(p_heat)),
            f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": int(round(p_cool.sum() * timestep / 3600)),
            f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": int(max(p_cool)),
        }

        df = pd.DataFrame(
            {
                f"{C.TEMP_IN}": temp_in.round(3),
                f"{Types.HEATING}{SEP}{C.LOAD}[W]": p_heat.round().astype(int),
                f"{Types.COOLING}{SEP}{C.LOAD}[W]": p_cool.round().astype(int),
            },
            index=data[O.WEATHER].index,
        )
        df.index.name = C.DATETIME

        return {"summary": summary, "timeseries": df}


def calculate_timeseries_1r1c(obj: dict, data: dict, timestep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate HVAC time series using a 1R1C model.

    Integrates the lumped-capacitance ODE

        C · dT/dt = G_tot · (T_ss − T) + P_h − P_c ,
        G_tot = 1/R + H_ve ,
        T_ss  = T_out + (G_int + G_sol) / G_tot ,

    with the **analytical exponential update** for each step under piecewise-constant
    forcings. This solution is exact (up to float32 round-off) for constant
    T_out/H_ve/gains over one step and is unconditionally stable for any Δt/τ,
    unlike the explicit-Euler scheme it replaces.

    The controller inverts the analytical update to pick the minimum heating
    or cooling power that lands the next-step temperature exactly on the
    active setpoint, then clips to the device power limit.

    Args:
        obj (dict): Building parameters (R, C, setpoints, power limits, flags).
        data (dict): Time-series inputs (weather, gains, ventilation conductance).
        timestep (float): Simulation step in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (indoor_temperature, heating_power, cooling_power).
    """
    # Get objects
    thermal_resistance = np.float32(obj[O.RESISTANCE])
    thermal_capacitance = np.float32(obj[O.CAPACITANCE])
    temp_init = np.float32(obj[O.TEMP_INIT])
    temp_min = np.float32(obj[O.TEMP_MIN])
    temp_max = np.float32(obj[O.TEMP_MAX])
    active_heat = bool(obj[O.ACTIVE_HEATING])
    active_cool = bool(obj[O.ACTIVE_COOLING])
    power_heat_max = np.float32(obj[O.POWER_HEATING])
    power_cool_max = np.float32(obj[O.POWER_COOLING])

    # Get data
    weather = data[O.WEATHER]
    temp_air = weather[C.TEMP_AIR].to_numpy(dtype=np.float32, copy=False)
    solar_gains = data[O.GAINS_SOLAR].to_numpy(dtype=np.float32, copy=False).ravel()
    internal_gains = data[O.GAINS_INTERNAL].to_numpy(dtype=np.float32, copy=False).ravel()
    ventilation = data[O.VENTILATION].to_numpy(dtype=np.float32, copy=False).ravel()

    n_steps = temp_air.shape[0]
    temp_in = np.empty(n_steps, dtype=np.float32)
    p_heat = np.zeros(n_steps, dtype=np.float32)
    p_cool = np.zeros(n_steps, dtype=np.float32)

    temp_in[0] = temp_init

    # Precompute loop invariants
    inv_resistance = np.float32(1.0) / thermal_resistance
    dt = np.float32(timestep)
    dt_over_cap = dt / thermal_capacitance

    temp_prev = temp_in[0]

    for t in range(1, n_steps):
        g_tot, decay, gain = calc_step_dynamics(ventilation[t], inv_resistance, dt, thermal_capacitance, dt_over_cap)
        t_ss_pas = calc_passive_steady_state(temp_air[t], solar_gains[t], internal_gains[t], g_tot, dt_over_cap)
        t_pas_next = calc_passive_next_temp(temp_prev, t_ss_pas, decay)

        p_heat[t] = calc_heating_power(t_pas_next, temp_min, gain, power_heat_max, active_heat)
        p_cool[t] = calc_cooling_power(t_pas_next, temp_max, gain, power_cool_max, active_cool)

        temp_prev = t_pas_next + gain * (p_heat[t] - p_cool[t])
        temp_in[t] = temp_prev

    return temp_in, p_heat, p_cool


# Below the small-G_tot threshold the analytical form (1 − exp(−x))/G_tot
# becomes numerically 0/0. Envelope conductances for real buildings are
# ≥ ~1 W/K, so this branch is defensive only.
_G_TOT_EPS = np.float32(1e-9)


def calc_step_dynamics(
    ventilation: float,
    inv_resistance: float,
    dt: float,
    capacitance: float,
    dt_over_cap: float,
) -> tuple[float, float, float]:
    """Return (G_tot, decay, gain) for one time step.

    - G_tot [W/K]: 1/R + H_ve
    - decay [-]: exp(-Δt · G_tot / C)  — 1R1C impulse response
    - gain [K/W]: (1 − decay) / G_tot  — sensitivity of T_next to (P_h − P_c)

    In the small-G_tot limit the ODE degenerates into a pure integrator;
    gain → Δt/C, decay → 1. Handled explicitly to avoid 0/0.
    """
    g_tot = inv_resistance + ventilation
    if g_tot > _G_TOT_EPS:
        x = dt * g_tot / capacitance
        one_minus_decay = -np.expm1(-x)  # numerically stable for small x
        decay = np.float32(1.0) - one_minus_decay
        gain = one_minus_decay / g_tot
    else:
        decay = np.float32(1.0)
        gain = dt_over_cap
    return g_tot, decay, gain


def calc_passive_steady_state(
    temp_air: float,
    solar_gains: float,
    internal_gains: float,
    g_tot: float,
    dt_over_cap: float,
) -> float:
    """Passive steady-state temperature T_ss (no active heating/cooling)."""
    if g_tot > _G_TOT_EPS:
        return temp_air + (solar_gains + internal_gains) / g_tot
    # Pure integrator: no finite steady state; return T_air so the caller's
    # analytical update collapses to a Euler step driven by gains alone.
    return temp_air


def calc_passive_next_temp(temp_prev: float, t_ss_pas: float, decay: float) -> float:
    """Next-step temperature under passive forcing only."""
    return t_ss_pas + (temp_prev - t_ss_pas) * decay


def calc_heating_power(
    t_pas_next: float,
    temp_min: float,
    gain: float,
    power_heat_max: float,
    active: bool,
) -> float:
    """Analytical inversion: minimum P_h that lifts T_next to T_min."""
    if not active:
        return 0
    if t_pas_next >= temp_min:
        return 0
    required = (temp_min - t_pas_next) / gain
    return required if required < power_heat_max else power_heat_max


def calc_cooling_power(
    t_pas_next: float,
    temp_max: float,
    gain: float,
    power_cool_max: float,
    active: bool,
) -> float:
    """Analytical inversion: minimum P_c that pulls T_next down to T_max."""
    if not active:
        return 0
    if t_pas_next <= temp_max:
        return 0
    required = (t_pas_next - temp_max) / gain
    return required if required < power_cool_max else power_cool_max
