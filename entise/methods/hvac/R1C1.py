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

        # Timestep in seconds (assuming a regular time grid). Compute from
        # just the first two timestamps — casting the whole column via
        # `.values.astype("datetime64[ns]")` costs ~20 ms/object on a
        # timezone-aware DatetimeIndex because pandas materializes the object
        # dtype first. Only the delta matters.
        dt_col = data[O.WEATHER][C.DATETIME]
        timestep = np.float32((dt_col.iloc[1] - dt_col.iloc[0]).total_seconds())

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
        # Weather and timestep — see note in generate() for why we avoid
        # materializing the full DATETIME column via astype.
        weather = data[O.WEATHER]
        index = weather.index
        dt_col = weather[C.DATETIME]
        dt_s = float((dt_col.iloc[1] - dt_col.iloc[0]).total_seconds())

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
        # NB: use ndarray.max() (C-level, single pass over the buffer), not
        # builtins.max() which iterates 8760 elements as Python objects.
        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": int(round(p_heat.sum() * timestep / 3600)),
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": int(p_heat.max()),
            f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": int(round(p_cool.sum() * timestep / 3600)),
            f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": int(p_cool.max()),
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


# Below the small-G_tot threshold the analytical form (1 − exp(−x))/G_tot
# becomes numerically 0/0. Envelope conductances for real buildings are
# ≥ ~1 W/K, so this branch is defensive only.
_G_TOT_EPS = np.float32(1e-9)


def calculate_timeseries_1r1c(obj: dict, data: dict, timestep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to numpy or numba path based on the active accelerator.

    The two paths implement the same physics (analytical exponential update)
    and produce numerically identical results up to a few ULPs. See
    :func:`_calculate_timeseries_numpy` for the physics and
    :mod:`entise.methods.hvac._R1C1_numba` for the JIT-compiled variant.

    The accelerator is chosen by :func:`entise.get_accelerator`, controlled
    by the ``ENTISE_ACCELERATOR`` environment variable and the
    :func:`entise.set_accelerator` runtime API.
    """
    from entise.perf import get_accelerator

    if get_accelerator() == "numba":
        # Lazy import: numba is optional (`pip install entise[numba]`).
        from entise.methods.hvac._R1C1_numba import calculate_timeseries_1r1c as _numba

        return _numba(obj, data, timestep)
    return _calculate_timeseries_numpy(obj, data, timestep)


def _calculate_timeseries_numpy(obj: dict, data: dict, timestep: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    power_heat_max = np.float32(obj[O.POWER_HEATING])
    power_cool_max = np.float32(obj[O.POWER_COOLING])
    inv_resistance = np.float32(1.0) / np.float32(obj[O.RESISTANCE])
    dt = np.float32(timestep)

    # Time-series inputs
    weather = data[O.WEATHER]
    temp_air = weather[C.TEMP_AIR].to_numpy(dtype=np.float32, copy=False)
    solar_gains = data[O.GAINS_SOLAR].to_numpy(dtype=np.float32, copy=False).ravel()
    internal_gains = data[O.GAINS_INTERNAL].to_numpy(dtype=np.float32, copy=False).ravel()
    ventilation = data[O.VENTILATION].to_numpy(dtype=np.float32, copy=False).ravel()

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

    n_steps = temp_air.shape[0]
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
    # The controllers invert the linear update to land T_next on the active
    # setpoint (T_min for heating, T_max for cooling), then clip to P_max.
    temp_prev = temp_in[0]
    for t in range(1, n_steps):
        d = decay[t]
        g = gain[t]
        t_ss = t_ss_pas[t]
        t_pas = t_ss + (temp_prev - t_ss) * d

        p_h = np.float32(0.0)
        if active_heat and t_pas < temp_min:
            need = (temp_min - t_pas) / g
            p_h = need if need < power_heat_max else power_heat_max

        p_c = np.float32(0.0)
        if active_cool and t_pas > temp_max:
            need = (t_pas - temp_max) / g
            p_c = need if need < power_cool_max else power_cool_max

        p_heat[t] = p_h
        p_cool[t] = p_c
        temp_prev = t_pas + g * (p_h - p_c)
        temp_in[t] = temp_prev

    return temp_in, p_heat, p_cool
