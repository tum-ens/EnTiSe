import logging
from typing import Optional, Tuple


import numpy as np
import pandas as pd


from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Constants as Const
from entise.constants import Objects as O
from entise.core.base import Method
from entise.core.utils import resolve_ts_or_scalar
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.methods.auxiliary.ventilation.selector import Ventilation
from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries
from entise.methods.hvac.defaults import (
DEFAULT_ACTIVE_COOLING,
DEFAULT_ACTIVE_HEATING,
DEFAULT_ACTIVE_INTERNAL_GAINS,
DEFAULT_ACTIVE_SOLAR_GAINS,
DEFAULT_ACTIVE_VENTILATION,
DEFAULT_FRAC_CONV_INTERNAL,
DEFAULT_FRAC_RAD_AW,
DEFAULT_POWER_COOLING,
DEFAULT_POWER_HEATING,
DEFAULT_SIGMA_7R2C_AW,
DEFAULT_SIGMA_7R2C_IW,
DEFAULT_TEMP_INIT,
DEFAULT_TEMP_MAX,
DEFAULT_TEMP_MIN,
DEFAULT_T_EQ_ALPHA_SW,
DEFAULT_T_EQ_H_O,
DEFAULT_VENTILATION,
DEFAULT_VENTILATION_SPLIT,

)

logger = logging.getLogger(__name__)

_WEATHER_CACHE: dict[tuple, pd.DataFrame] = {}


class R7C2(Method):

    """TODO: Docstring for 7R2C model."""

    types = [Types.HVAC]
    name = "7R2C"

    # Required RC & weather
    required_keys = [
    O.R_1_AW, O.C_1_AW,
    O.R_1_IW, O.C_1_IW,
    O.R_ALPHA_STAR_IL, O.R_ALPHA_STAR_AW, O.R_ALPHA_STAR_IW,
    O.R_REST_AW,
    O.WEATHER,
    ]

    # Optional controls, splits, and auxiliaries
    optional_keys = [
    # Controls
    O.POWER_HEATING, O.POWER_COOLING,
    O.ACTIVE_HEATING, O.ACTIVE_COOLING,
    O.ACTIVE_GAINS_INTERNAL, O.ACTIVE_GAINS_SOLAR, O.ACTIVE_VENTILATION,
    O.TEMP_INIT, O.TEMP_MIN, O.TEMP_MAX, O.TEMP_SUPPLY,
    # Geometry
    O.AREA, O.HEIGHT,
    # Gains splits
    O.FRAC_CONV_INTERNAL, O.FRAC_RAD_AW, # IW is derived
    # HVAC sigma as three scalars (AW, IW, conv is derived)
    O.SIGMA_7R2C_AW, O.SIGMA_7R2C_IW,
    # Ventilation
    O.H_VE, O.VENTILATION_SPLIT,
    # Equivalent outdoor (sol-air)
    O.T_EQ, O.T_EQ_COL, O.T_EQ_ALPHA_SW, O.T_EQ_H_O,
    # Timeseries
    O.GAINS_INTERNAL, O.GAINS_SOLAR, O.VENTILATION
    ]

    required_timeseries = [O.WEATHER]
    optional_timeseries = [O.WINDOWS, O.GAINS_INTERNAL, O.GAINS_SOLAR, O.H_VE, O.T_EQ] # Hint set of common auxiliaries

    output_summary = {
    f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": "total heating demand",
    f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": "maximum heating load",
    f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": "total cooling demand",
    f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": "maximum cooling load",
    }

    output_timeseries = {
    C.TEMP_IN: "indoor air temperature",
    f"{Types.HEATING}{SEP}{C.LOAD}[W]": "heating load",
    f"{Types.COOLING}{SEP}{C.LOAD}[W]": "cooling load",
    }


    def generate(self, obj: dict = None, data: dict = None, ts_type: str = Types.HVAC, **kwargs) -> dict:
        obj, data = self._process_kwargs(obj, data, **kwargs)
        obj, data = self._get_input_data(obj, data, ts_type)
        data = self._prepare_inputs(obj, data)

        temp_in, p_heat, p_cool = calculate_timeseries_7r2c(**data)

        meta = data['meta']
        return self._format_output(
        temp_in.round(3), p_heat.round().astype(int), p_cool.round().astype(int), meta["index"], meta["dt_s"]
        )


    def _get_input_data(self, obj: dict, data: dict, method_type: str = Types.HVAC) -> tuple[dict, dict]:
        """
        Collect raw inputs for this method:
        - merge method-specific overrides (e.g. 'hvac:TEMP_MIN')
        - validate required keys exist
        - set defaults for missing optional scalar keys
        - pick the relevant timeseries tables
        - normalize weather dataframe

        No interpretation of columns, no unit conversion, no time-series generation.
        """
        obj_out = self._get_relevant_objects(obj, method_type)
        data_out = self._prepare_data_tables(obj, data, method_type)
        return obj_out, data_out


    def _prepare_inputs(self, obj: dict, data: dict) -> dict:

        data = self._prepare_weather_cache(obj, data)

        # Add data for computation
        data_out = {}

        # Create meta data
        weather = data[O.WEATHER]
        index = weather.index
        meta = {
            O.WEATHER: weather,
            "index": index,
            "dt_s": float((index[1] - index[0]) / np.timedelta64(1, "s")),
        }
        data_out['meta'] = meta

        # Gains (generate if active and missing)
        data_out['series'] = {}
        gains = self._compute_gains(obj, data)
        data_out['series']['gains'] = gains

        # Ventilation → split into mechanical & infiltration
        ven_df = VentilationTimeSeries().generate(obj, data).squeeze() # Calls VDI 6007 conform norm directly
        vent_split = resolve_ts_or_scalar(obj, data, O.VENTILATION_SPLIT, index, default=DEFAULT_VENTILATION_SPLIT)
        Hve_vent_series = ven_df * vent_split
        Hve_inf_series = ven_df * (1.0 - vent_split)
        ventilation = pd.concat([Hve_vent_series, Hve_inf_series], axis=1, keys=["Hve_vent", "Hve_inf"])
        data_out['series']['ventilation'] = ventilation

        # Equivalent outdoor temperature series (T_eq)
        data_out['series']['T_eq'] = self._calc_teq_series(obj, data, weather, index)

        # Controls
        temp_sup = obj.get(O.TEMP_SUPPLY, None)
        temp_sup = resolve_ts_or_scalar(obj, data, O.TEMP_SUPPLY, index) if temp_sup is not None else weather[C.TEMP_AIR]
        controls = {
            O.TEMP_MIN: resolve_ts_or_scalar(obj, data, O.TEMP_MIN, index, default=DEFAULT_TEMP_MIN),
            O.TEMP_MAX: resolve_ts_or_scalar(obj, data, O.TEMP_MAX, index, default=DEFAULT_TEMP_MAX),
            O.TEMP_SUPPLY: temp_sup,
            O.POWER_HEATING: resolve_ts_or_scalar(obj, data, O.POWER_HEATING, index, default=DEFAULT_POWER_HEATING),
            O.POWER_COOLING: resolve_ts_or_scalar(obj, data, O.POWER_COOLING, index, default=DEFAULT_POWER_COOLING),
            O.ACTIVE_HEATING: resolve_ts_or_scalar(obj, data, O.ACTIVE_HEATING, index, default=DEFAULT_ACTIVE_HEATING),
            O.ACTIVE_COOLING: resolve_ts_or_scalar(obj, data, O.ACTIVE_COOLING, index, default=DEFAULT_ACTIVE_COOLING),
        }
        controls = pd.DataFrame(controls)
        data_out['controls'] = controls

        # Parameters
        volume = float(obj.get(O.AREA, Const.DEFAULT_AREA.value)) * float(obj.get(O.HEIGHT, Const.DEFAULT_HEIGHT.value))
        rho_air = 1.2 # kg/m3
        cp_air = 1000.0 # J/kgK
        capacity_air = volume * rho_air * cp_air
        params = {
            O.TEMP_INIT: float(obj.get(O.TEMP_INIT, DEFAULT_TEMP_INIT)),
            O.R_1_AW: float(obj[O.R_1_AW]),
            O.C_1_AW: float(obj[O.C_1_AW]),
            O.R_1_IW: float(obj[O.R_1_IW]),
            O.C_1_IW: float(obj[O.C_1_IW]),
            O.R_ALPHA_STAR_IL: float(obj[O.R_ALPHA_STAR_IL]),
            O.R_ALPHA_STAR_AW: float(obj[O.R_ALPHA_STAR_AW]),
            O.R_ALPHA_STAR_IW: float(obj[O.R_ALPHA_STAR_IW]),
            O.R_REST_AW: float(obj[O.R_REST_AW]),
            O.CAPACITANCE_AIR: float(capacity_air),
        }
        data_out['params'] = params

        # Splits (normalize defensively)
        sigma_aw = float(obj.get(O.SIGMA_7R2C_AW, DEFAULT_SIGMA_7R2C_AW))
        sigma_iw = float(obj.get(O.SIGMA_7R2C_IW, DEFAULT_SIGMA_7R2C_IW))
        f_aw = float(obj.get(O.FRAC_RAD_AW, DEFAULT_FRAC_RAD_AW))
        f_aw = min(max(f_aw, 0.0), 1.0)
        splits = {
            O.SIGMA_7R2C_AW: sigma_aw,
            O.SIGMA_7R2C_IW: sigma_iw,
            "sigma_conv": max(0.0, 1.0 - (sigma_aw + sigma_iw)),
            O.FRAC_CONV_INTERNAL: float(obj.get(O.FRAC_CONV_INTERNAL, DEFAULT_FRAC_CONV_INTERNAL)),
            O.FRAC_RAD_AW: f_aw,
            "f_iw": 1.0 - f_aw
        }
        data_out['splits'] = splits

        return data_out

    @staticmethod
    def _format_output(temp_in: pd.Series, p_heat: pd.Series, p_cool: pd.Series, index: pd.DatetimeIndex, dt_s: float) -> dict:
        E_h_Wh = float(np.trapezoid(p_heat.values, dx=dt_s) / 3600.0)
        E_c_Wh = float(np.trapezoid(p_cool.values, dx=dt_s) / 3600.0)
        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": E_h_Wh,
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": float(max(p_heat.max(), 0.0)),
            f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": E_c_Wh,
            f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": float(max(p_cool.max(), 0.0)),
        }

        df = pd.DataFrame({
        C.TEMP_IN: temp_in,
            f"{Types.HEATING}{SEP}{C.LOAD}[W]": p_heat,
            f"{Types.COOLING}{SEP}{C.LOAD}[W]": p_cool,
        }, index=index)
        df.index.name = C.DATETIME

        return {"summary": summary, "timeseries": df}


    def _get_relevant_objects(self, obj: dict, method_type: str = Types.HVAC) -> dict:
        # TODO: Change this so that it takes the data from self.objects and make this for loops.
        #   This probably also means changing the lists to dicts and having default values as the value
        obj_out = {
        O.ID: self.get_with_backup(obj, O.ID),
        O.LAT: self.get_with_method_backup(obj, O.LAT, method_type),
        O.LON: self.get_with_method_backup(obj, O.LON, method_type),
        # Geometry (for air capacity)
        O.AREA: self.get_with_method_backup(obj, O.AREA, method_type, Const.DEFAULT_AREA.value),
        O.HEIGHT: self.get_with_method_backup(obj, O.HEIGHT, method_type, Const.DEFAULT_HEIGHT.value),
        # Controls
        O.ACTIVE_HEATING: self.get_with_method_backup(obj, O.ACTIVE_HEATING, method_type, DEFAULT_ACTIVE_HEATING),
        O.ACTIVE_COOLING: self.get_with_method_backup(obj, O.ACTIVE_COOLING, method_type, DEFAULT_ACTIVE_COOLING),
        O.ACTIVE_GAINS_INTERNAL: self.get_with_method_backup(obj, O.ACTIVE_GAINS_INTERNAL, method_type,
                                                             DEFAULT_ACTIVE_INTERNAL_GAINS),
        O.ACTIVE_GAINS_SOLAR: self.get_with_method_backup(obj, O.ACTIVE_GAINS_SOLAR, method_type,
                                                          DEFAULT_ACTIVE_SOLAR_GAINS),
        O.ACTIVE_VENTILATION: self.get_with_method_backup(obj, O.ACTIVE_VENTILATION, method_type,
                                                          DEFAULT_ACTIVE_VENTILATION),
        O.POWER_HEATING: self.get_with_method_backup(obj, O.POWER_HEATING, method_type, DEFAULT_POWER_HEATING),
        O.POWER_COOLING: self.get_with_method_backup(obj, O.POWER_COOLING, method_type, DEFAULT_POWER_COOLING),
        O.TEMP_INIT: self.get_with_method_backup(obj, O.TEMP_INIT, method_type, DEFAULT_TEMP_INIT),
        O.TEMP_MIN: self.get_with_method_backup(obj, O.TEMP_MIN, method_type, DEFAULT_TEMP_MIN),
        O.TEMP_MAX: self.get_with_method_backup(obj, O.TEMP_MAX, method_type, DEFAULT_TEMP_MAX),
        O.TEMP_SUPPLY: self.get_with_method_backup(obj, O.TEMP_SUPPLY, method_type),
        # RC parameters (7R2C)
        O.R_1_AW: self.get_with_method_backup(obj, O.R_1_AW, method_type),
        O.C_1_AW: self.get_with_method_backup(obj, O.C_1_AW, method_type),
        O.R_1_IW: self.get_with_method_backup(obj, O.R_1_IW, method_type),
        O.C_1_IW: self.get_with_method_backup(obj, O.C_1_IW, method_type),
        O.R_ALPHA_STAR_IL: self.get_with_method_backup(obj, O.R_ALPHA_STAR_IL, method_type),
        O.R_ALPHA_STAR_AW: self.get_with_method_backup(obj, O.R_ALPHA_STAR_AW, method_type),
        O.R_ALPHA_STAR_IW: self.get_with_method_backup(obj, O.R_ALPHA_STAR_IW, method_type),
        O.R_REST_AW: self.get_with_method_backup(obj, O.R_REST_AW, method_type),
        # Gains & splits
        O.FRAC_CONV_INTERNAL: self.get_with_method_backup(obj, O.FRAC_CONV_INTERNAL, method_type,
                                                          DEFAULT_FRAC_CONV_INTERNAL),
        O.FRAC_RAD_AW: self.get_with_method_backup(obj, O.FRAC_RAD_AW, method_type, DEFAULT_FRAC_RAD_AW),
        O.SIGMA_7R2C_AW: self.get_with_method_backup(obj, O.SIGMA_7R2C_AW, method_type, DEFAULT_SIGMA_7R2C_AW),
        O.SIGMA_7R2C_IW: self.get_with_method_backup(obj, O.SIGMA_7R2C_IW, method_type, DEFAULT_SIGMA_7R2C_IW),
        # Ventilation
        O.VENTILATION: self.get_with_method_backup(obj, O.VENTILATION, method_type, DEFAULT_VENTILATION),
        O.VENTILATION_COL: self.get_with_method_backup(obj, O.VENTILATION_COL, method_type),
        O.VENTILATION_SPLIT: self.get_with_method_backup(obj, O.VENTILATION_SPLIT, method_type,
                                                         DEFAULT_VENTILATION_SPLIT),
        # Equivalent outdoor temperature (sol-air)
        O.T_EQ: self.get_with_method_backup(obj, O.T_EQ, method_type),
        O.T_EQ_COL: self.get_with_method_backup(obj, O.T_EQ_COL, method_type),
        O.T_EQ_ALPHA_SW: self.get_with_method_backup(obj, O.T_EQ_ALPHA_SW, method_type, DEFAULT_T_EQ_ALPHA_SW),
        O.T_EQ_H_O: self.get_with_method_backup(obj, O.T_EQ_H_O, method_type, DEFAULT_T_EQ_H_O),
        # Timeseries references
        O.WEATHER: self.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER),
        O.GAINS_INTERNAL: self.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type),
        O.GAINS_INTERNAL_COL: self.get_with_method_backup(obj, O.GAINS_INTERNAL_COL, method_type),
        O.GAINS_SOLAR: self.get_with_method_backup(obj, O.GAINS_SOLAR, method_type),
        }

        # Future possible code
        # obj_out = {}
        # for key in self.required_keys:
        # value = obj_out.get(key)
        # if value is None:
        # raise ValueError(f"Required key '{key}' not found for method '{method_type}'.")
        # obj_out[key] = value
        # for key in self.optional_keys:
        # obj_out[key] = obj_out.get(key)

        return {k: v for k, v in obj_out.items() if v is not None}


    def _prepare_data_tables(self, obj: dict, data: dict, method_type: str = Types.HVAC) -> dict:
        data_out = {}

        for key in self.required_timeseries:
            ts_key = self.get_with_method_backup(obj, key, method_type, key)
            ts_data = self.get_with_backup(data, ts_key)
            if ts_data is None:
                raise ValueError(f"Required timeseries key '{ts_key}' for '{key}' not found in data.")
        data_out[key] = ts_data

        for key in self.optional_timeseries:
            ts_key = self.get_with_method_backup(obj, key, method_type)
            ts_data = self.get_with_backup(data, ts_key)
            if ts_data is not None:
                data_out[key] = ts_data

        return {k: v for k, v in data_out.items() if v is not None}


    def _prepare_weather_cache(self, obj: dict, data: dict, method_type: str = Types.HVAC) -> dict:
        weather_key = self.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather_cached = _WEATHER_CACHE.get(weather_key)
        if weather_cached is None and O.WEATHER in data:
            w = data[O.WEATHER].copy()
            w = self._strip_weather_height(w)
            w[C.DATETIME] = pd.to_datetime(w[C.DATETIME])
            w.set_index(C.DATETIME, inplace=True, drop=False)
            data[O.WEATHER] = w
            _WEATHER_CACHE[weather_key] = w
        elif weather_cached is not None:
            data[O.WEATHER] = weather_cached

        return data


    @staticmethod
    def _compute_gains(obj: dict, data: dict) -> pd.DataFrame:
        g_int_df = InternalGains().generate(obj, data)
        g_sol_df = SolarGains().generate(obj, data)

        return pd.concat([g_int_df, g_sol_df], axis=1, keys=["g_int", "g_sol"])

    def _calc_teq_series(self, obj: dict, data: dict, weather: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        teq = obj.get(O.T_EQ)
        if teq is not None:
            teq_series = resolve_ts_or_scalar(obj, data, O.T_EQ, index)
        else:
            # 1. Calculate Base Sol-Air Temperature (Assume this applies to Opaque parts)
            alpha_sw = float(obj.get(O.T_EQ_ALPHA_SW, DEFAULT_T_EQ_ALPHA_SW))
            h_o = float(obj.get(O.T_EQ_H_O, DEFAULT_T_EQ_H_O))
            if C.SOLAR_GHI not in weather.columns:
                raise ValueError(f"Weather data must contain '{C.SOLAR_GHI}' column.")

            ghi = weather.get(C.SOLAR_GHI, None).astype(float)
            teq_opaque = weather[C.TEMP_AIR].astype(float) + alpha_sw * ghi / max(h_o, 1e-6)

            # 2. Reverse Engineer Conductances to weight the temperature
            # Get Total Resistance from RC parameters
            r1_aw = float(obj.get(O.R_1_AW))
            r_rest_aw = float(obj.get(O.R_REST_AW))

            # Total Conductance (H_tot) ~ 1 / R_total
            # (Using a small epsilon to avoid division by zero if R is 0)
            h_tot = 1.0 / max((r1_aw + r_rest_aw), 1e-6)

            # Calculate Window Conductance (H_win) from available window data
            h_win = 0.0
            if O.WINDOWS in data:
                windows_df = data[O.WINDOWS]
                if not windows_df.empty:
                    for _, row in windows_df.iterrows():
                        u_val = float(row.get(C.U_VALUE, 0.0))
                        area = float(row.get(C.AREA, 0.0))
                        h_win += u_val * area

            # Derive Opaque Conductance (H_op)
            h_op = max(0.0, h_tot - h_win)

            # 3. Calculate Weighted T_eq
            # Formula: (T_sol_air * H_op + T_air * H_win) / H_tot
            if h_tot > 1e-6:
                t_air = weather[C.TEMP_AIR].astype(float)
                teq_series = (teq_opaque * h_op + t_air * h_win) / h_tot
            else:
                teq_series = teq_opaque
        return pd.DataFrame({"T_eq": teq_series}, index=index)


def calculate_timeseries_7r2c(meta: dict, series: dict, controls: pd.DataFrame, params: dict, splits: dict) \
        -> tuple[pd.Series, pd.Series, pd.Series]:
    """Time loop with free-float → setpoint/clamp logic (skeleton)."""

    idx: pd.DatetimeIndex = meta["index"]
    dt_s: float = float(meta["dt_s"])
    weather: pd.DataFrame = meta[O.WEATHER]

    n_len = len(idx)

    # Weather
    T_out_arr = weather[C.TEMP_AIR].to_numpy(dtype=float)
    T_eq_arr = series["T_eq"]["T_eq"].to_numpy(dtype=float)

    # Gains
    g_int_arr = series["gains"]["g_int"].to_numpy(dtype=float)
    g_sol_arr = series["gains"]["g_sol"].to_numpy(dtype=float)

    # Ventilation
    Hve_vent_arr = series["ventilation"]["Hve_vent"].to_numpy(dtype=float)
    Hve_inf_arr = series["ventilation"]["Hve_inf"].to_numpy(dtype=float)

    # Controls
    T_min_arr = controls[O.TEMP_MIN].to_numpy(dtype=float)
    T_max_arr = controls[O.TEMP_MAX].to_numpy(dtype=float)
    T_sup_arr = controls[O.TEMP_SUPPLY].to_numpy(dtype=float)

    P_heat_arr = controls[O.POWER_HEATING].to_numpy(dtype=float)
    P_cool_arr = controls[O.POWER_COOLING].to_numpy(dtype=float)

    on_heat_arr = controls[O.ACTIVE_HEATING].to_numpy(dtype=bool)
    on_cool_arr = controls[O.ACTIVE_COOLING].to_numpy(dtype=bool)

    # Parameters
    T_init = params[O.TEMP_INIT]
    R_1_aw = params[O.R_1_AW]
    C_1_aw = params[O.C_1_AW]
    R_1_iw = params[O.R_1_IW]
    C_1_iw = params[O.C_1_IW]
    R_alpha_star_il = params[O.R_ALPHA_STAR_IL]
    R_alpha_star_aw = params[O.R_ALPHA_STAR_AW]
    R_alpha_star_iw = params[O.R_ALPHA_STAR_IW]
    R_rest_aw = params[O.R_REST_AW]
    C_air = params[O.CAPACITANCE_AIR]

    # Splits
    sigma_aw = splits[O.SIGMA_7R2C_AW]
    sigma_iw = splits[O.SIGMA_7R2C_IW]
    sigma_conv = splits["sigma_conv"]
    f_conv_int = splits[O.FRAC_CONV_INTERNAL]
    f_rad_aw = splits[O.FRAC_RAD_AW]
    f_rad_iw = splits["f_iw"]

    # Compute LUE resistances and inverses
    inv_R_1_aw = 1.0 / R_1_aw
    inv_R_1_iw = 1.0 / R_1_iw
    inv_R_a_il = 1.0 / R_alpha_star_il
    inv_R_a_aw = 1.0 / R_alpha_star_aw
    inv_R_a_iw = 1.0 / R_alpha_star_iw
    inv_R_r_aw = 1.0 / R_rest_aw

    # Initial states (air and two masses)
    Ta = T_init
    Tm_aw = (3 * T_init - T_out_arr[0]) / 4  # Simplified initial guess
    Tm_iw = T_init

    T_in_arr = np.empty(n_len, dtype=float)
    Qh_arr = np.empty(n_len, dtype=float)
    Qc_arr = np.empty(n_len, dtype=float)

    T_in_arr[0] = Ta
    Qh_arr[0] = 0.0
    Qc_arr[0] = 0.0

    # Split gains → (conv, aw, iw)
    phi_conv_arr, phi_aw_arr, phi_iw_arr = split_gains_7r2c(g_int_arr, g_sol_arr, f_conv_int, f_rad_aw, f_rad_iw)

    # Precompute optimization constants for constant ventilation case
    Y_tset_const = calc_y_tset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                                     C_1_aw, C_1_iw, sigma_aw, sigma_iw, sigma_conv, dt_s)
    if np.allclose(Hve_vent_arr, Hve_vent_arr[0]) and np.allclose(Hve_inf_arr, Hve_inf_arr[0]):
        Y_phiset_const = calc_y_phiset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                                             C_1_aw, C_1_iw, C_air, Hve_inf_arr[0], Hve_vent_arr[0], dt_s)
    else:
        Y_phiset_const = None


    for t in range(1, n_len):
        T_out = T_out_arr[t]
        T_sup = T_sup_arr[t]
        T_eq = float(T_eq_arr[t])

        Hve_vent = float(Hve_vent_arr[t])
        Hve_inf = float(Hve_inf_arr[t])

        phi_conv = float(phi_conv_arr[t])
        phi_aw = float(phi_aw_arr[t])
        phi_iw = float(phi_iw_arr[t])

        T_min = float(T_min_arr[t])
        T_max = float(T_max_arr[t])

        on_heat = bool(on_heat_arr[t])
        on_cool = bool(on_cool_arr[t])

        P_h_max = float(P_heat_arr[t])
        P_c_max = float(P_cool_arr[t])

        # 1) Free-float (Q_hc=0)
        sol_free = solve_step_phiset_7r2c_optim(
            inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
            C_1_aw, C_1_iw, C_air, Hve_vent, Hve_inf,
            T_out, T_eq, T_sup, Ta, Tm_aw, Tm_iw,
            phi_conv, phi_aw, phi_iw, 0.0, 0.0, 0.0, 0.0, dt_s, Y_phiset_const)
        Ta_free = float(sol_free[3]); Tm_aw_free = float(sol_free[0]); Tm_iw_free = float(sol_free[6])

        # 2) Check against setpoints and clamp if necessary
        Q = 0.0
        # Heating
        if Ta_free < T_min and on_heat:
            sol_tset = solve_step_tset_7r2c_optim(
                inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                C_1_aw, C_1_iw, C_air, sigma_aw, sigma_iw, sigma_conv, Hve_vent, Hve_inf,
                T_out, T_eq, T_sup, T_min, Ta, Tm_aw, Tm_iw, phi_conv, phi_aw, phi_iw, dt_s, Y_tset_const)
            Q_req = float(sol_tset[4])
            Q, clamped = clamp_power(Q_req, on_heat, on_cool, P_h_max, P_c_max)
            if clamped:
                Q_hk_iw = Q * sigma_iw
                Q_hk_aw = Q * sigma_aw
                Q_hk_kon = Q * sigma_conv
                sol_phi = solve_step_phiset_7r2c_optim(
                    inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                    C_1_aw, C_1_iw, C_air, Hve_vent, Hve_inf,
                    T_out, T_eq, T_sup, Ta, Tm_aw, Tm_iw,
                    phi_conv, phi_aw, phi_iw, Q, Q_hk_iw, Q_hk_aw, Q_hk_kon, dt_s, Y_phiset_const)
                Tm_aw, Ta, Tm_iw = float(sol_phi[0]), float(sol_phi[3]), float(sol_phi[6])
            else:
                Tm_aw, Ta, Tm_iw = float(sol_tset[0]), float(sol_tset[3]), float(sol_tset[6])
        # Cooling
        elif Ta_free > T_max and on_cool:
            sol_tset = solve_step_tset_7r2c_optim(
                inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                C_1_aw, C_1_iw, C_air, sigma_aw, sigma_iw, sigma_conv, Hve_vent, Hve_inf,
                T_out, T_eq, T_sup, T_max, Ta, Tm_aw, Tm_iw, phi_conv, phi_aw, phi_iw, dt_s, Y_tset_const)
            Q_req = float(sol_tset[4])
            Q, clamped = clamp_power(Q_req, on_heat, on_cool, P_h_max, P_c_max)
            if clamped:
                Q_hk_iw = Q * sigma_iw
                Q_hk_aw = Q * sigma_aw
                Q_hk_kon = Q * sigma_conv
                sol_phi = solve_step_phiset_7r2c_optim(
                    inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                    C_1_aw, C_1_iw, C_air, Hve_vent, Hve_inf,
                    T_out, T_eq, T_sup, Ta, Tm_aw, Tm_iw,
                    phi_conv, phi_aw, phi_iw, Q, Q_hk_iw, Q_hk_aw, Q_hk_kon, dt_s, Y_phiset_const)
                Tm_aw, Ta, Tm_iw = float(sol_phi[0]), float(sol_phi[3]), float(sol_phi[6])
            else:
                Tm_aw, Ta, Tm_iw = float(sol_tset[0]), float(sol_tset[3]), float(sol_tset[6])
        # No action needed, use free-float results
        else:
            Tm_aw, Ta, Tm_iw = Tm_aw_free, Ta_free, Tm_iw_free

        T_in_arr[t] = Ta
        Qh_arr[t] = max(Q, 0.0)
        Qc_arr[t] = max(-Q, 0.0)


    T_in = pd.Series(T_in_arr, index=idx, dtype=float)
    Q_heat = pd.Series(Qh_arr, index=idx, dtype=float)
    Q_cool = pd.Series(Qc_arr, index=idx, dtype=float)

    return T_in, Q_heat, Q_cool


def split_gains_7r2c(g_int: np.ndarray, g_sol: np.ndarray, f_conv: float, f_aw: float, f_iw: float) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised split of internal & solar gains into convective and radiant parts."""
    g_int = np.asarray(g_int, dtype=float)
    g_sol = np.asarray(g_sol, dtype=float)

    q_conv = f_conv * g_int
    q_rad_int = (1.0 - f_conv) * g_int
    q_rad_tot = q_rad_int + g_sol

    phi_aw = f_aw * q_rad_tot
    phi_iw = f_iw * q_rad_tot

    return q_conv, phi_aw, phi_iw


def clamp_power(Q_req: float, on_h: bool, on_c: bool, P_h_max: float, P_c_max: float) -> tuple[float, bool]:
    if Q_req >= 0.0:
        if not on_h:
            return 0.0, True
        Q = min(Q_req, P_h_max)
        return Q, (Q < Q_req)
    else:
        if not on_c:
            return 0.0, True
        Q = max(Q_req, -P_c_max)
    return Q, (Q > Q_req)


def solve_step_tset_7r2c(R_1_aw: float, R_1_iw: float,
        R_alpha_star_il: float, R_alpha_star_aw: float, R_alpha_star_iw: float, R_rest_aw: float,
        C_1_aw: float, C_1_iw: float, C_air: float,
        sigma_aw: float, sigma_iw: float, sigma_conv: float,
        Hve_vent: float, Hve_inf: float,
        T_out: float, T_eq: float, T_sup: float, T_set: float,
        Ta_prev: float, Tm_aw_prev: float, Tm_iw_prev: float,
        phi_conv: float, phi_aw: float, phi_iw: float,
        tau: float, Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """One-step 7R2C solve in setpoint mode (compute required HVAC load).

    Returns:
    np.ndarray: [Tm_aw, Ts_aw, T_lu_star, T_air=T_set, Q_hc_req, Ts_iw, Tm_iw]
    """
    if Y is None:
        Y = calc_y_tset(R_1_aw, R_1_iw, R_alpha_star_il, R_alpha_star_aw, R_alpha_star_iw, R_rest_aw,
                        C_1_aw, C_1_iw, sigma_aw, sigma_iw, sigma_conv, tau)

    q = calc_q_tset(R_rest_aw, R_alpha_star_il, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                    T_set, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                    phi_conv, phi_aw, phi_iw, tau)

    return solve_y_tset_7r2c(Y, q, T_set)


def solve_step_tset_7r2c_optim(inv_R_1_aw: float, inv_R_1_iw: float,
                            inv_R_a_il: float, inv_R_a_aw: float, inv_R_a_iw: float, inv_R_r_aw: float,
                            C_1_aw: float, C_1_iw: float, C_air: float,
                            sigma_aw: float, sigma_iw: float, sigma_conv: float,
                            Hve_vent: float, Hve_inf: float,
                            T_out: float, T_eq: float, T_sup: float, T_set: float,
                            Ta_prev: float, Tm_aw_prev: float, Tm_iw_prev: float,
                            phi_conv: float, phi_aw: float, phi_iw: float,
                            tau: float, Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    if Y is None:
        Y = calc_y_tset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                              C_1_aw, C_1_iw, sigma_aw, sigma_iw, sigma_conv, tau)

    q = calc_q_tset_optim(inv_R_r_aw, inv_R_a_il, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                          T_set, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                          phi_conv, phi_aw, phi_iw, tau)

    return solve_y_tset_7r2c(Y, q, T_set)


def solve_y_tset_7r2c(Y, q, T_set: float) -> np.ndarray:
    """Solve Y matrix for 7R2C setpoint mode given precomputed Y and q."""
    # y = [Tm_aw, Ts_aw, T_lu_star, Q_hc_req, Ts_iw, Tm_iw]
    y = np.linalg.solve(Y, q)
    return np.array([y[0], y[1], y[2], T_set, y[3], y[4], y[5]], dtype=float)


def solve_step_phiset_7r2c(R_1_aw: float, R_1_iw: float,
                        R_alpha_star_il: float, R_alpha_star_aw: float, R_alpha_star_iw: float, R_rest_aw: float,
                        C_1_aw: float, C_1_iw: float, C_air: float,
                        Hve_vent: float, Hve_inf: float,
                        T_out: float, T_eq: float, T_sup: float,
                        Ta_prev: float, Tm_aw_prev: float, Tm_iw_prev: float,
                        phi_conv: float, phi_aw: float, phi_iw: float,
                        Q_hc: float, Q_hk_iw: float, Q_hk_aw: float, Q_hk_kon: float,
                        tau: float, Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    """One-step 7R2C solve in applied‑load mode (compute resultant temperatures).

    Returns:
    np.ndarray: [Tm_aw, Ts_aw, T_lu_star, T_air, Q_hc, Ts_iw, Tm_iw]
    """
    if Y is None:
        Y = calc_y_phiset(R_1_aw, R_1_iw, R_alpha_star_il, R_alpha_star_aw, R_alpha_star_iw, R_rest_aw,
                          C_1_aw, C_1_iw, C_air, Hve_inf, Hve_vent, tau)

    q = calc_q_phiset(R_rest_aw, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                      Q_hk_aw, Q_hk_iw, Q_hk_kon, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                      phi_conv, phi_aw, phi_iw, tau)

    return solve_y_phiset_7r2c(Y, q, Q_hc)


def solve_step_phiset_7r2c_optim(inv_R_1_aw: float, inv_R_1_iw: float,
                                inv_R_a_il: float, inv_R_a_aw: float, inv_R_a_iw: float, inv_R_r_aw: float,
                                C_1_aw: float, C_1_iw: float, C_air: float,
                                Hve_vent: float, Hve_inf: float,
                                T_out: float, T_eq: float, T_sup: float,
                                Ta_prev: float, Tm_aw_prev: float, Tm_iw_prev: float,
                                phi_conv: float, phi_aw: float, phi_iw: float,
                                Q_hc: float, Q_hk_iw: float, Q_hk_aw: float, Q_hk_kon: float,
                                tau: float, Y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
    if Y is None:
        Y = calc_y_phiset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                                C_1_aw, C_1_iw, C_air, Hve_inf, Hve_vent,  tau)

    q = calc_q_phiset_optim(inv_R_r_aw, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                            Q_hk_aw, Q_hk_iw, Q_hk_kon, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                            phi_conv, phi_aw, phi_iw, tau)

    return solve_y_phiset_7r2c(Y, q, Q_hc)


def solve_y_phiset_7r2c(Y, q, Q_hc: float) -> np.ndarray:
    """Solve Y matrix for 7R2C applied‑load mode given precomputed Y and q."""
    # y = [Tm_aw, Ts_aw, T_lu_star, T_air, Ts_iw, Tm_iw]
    y = np.linalg.solve(Y, q)
    return np.array([y[0], y[1], y[2], y[3], Q_hc, y[4], y[5]], dtype=float)


def calc_y_tset(R_1_aw, R_1_iw, R_alpha_star_il, R_alpha_star_aw, R_alpha_star_iw, R_rest_aw,
                C_1_aw, C_1_iw, sigma_aw, sigma_iw, sigma_conv, tau):
    """Calculate Y matrix for 7R2C setpoint mode."""
    # Precompute conductances (1/R) — speeds things up
    inv_R_1_aw = 1.0 / R_1_aw
    inv_R_1_iw = 1.0 / R_1_iw
    inv_R_a_il = 1.0 / R_alpha_star_il
    inv_R_a_aw = 1.0 / R_alpha_star_aw
    inv_R_a_iw = 1.0 / R_alpha_star_iw
    inv_R_r_aw = 1.0 / R_rest_aw

    return calc_y_tset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                             C_1_aw, C_1_iw, sigma_aw, sigma_iw, sigma_conv, tau)


def calc_y_tset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                      C_1_aw, C_1_iw, sigma_aw, sigma_iw, sigma_conv, tau):
    """Calculate Y matrix for 7R2C setpoint mode in optimized way."""
    # Build Y (6x6)
    Y = np.zeros((6, 6), dtype=np.float64)

    # Row 0 (theta_m_aw)
    Y[0, 0] = -(inv_R_r_aw + inv_R_1_aw) - C_1_aw / tau
    Y[0, 1] = inv_R_1_aw

    # Row 1 (theta_s_aw)
    Y[1, 0] = inv_R_1_aw
    Y[1, 1] = -(inv_R_1_aw + inv_R_a_aw)
    Y[1, 2] = inv_R_a_aw
    Y[1, 3] = sigma_aw

    # Row 2 (theta_lu_star)
    Y[2, 1] = inv_R_a_aw
    Y[2, 2] = -(inv_R_a_aw + inv_R_a_il + inv_R_a_iw)
    Y[2, 4] = inv_R_a_iw

    # Row 3 (air node balance uses Q_hc unknown, location col=3)
    Y[3, 2] = inv_R_a_il
    Y[3, 3] = sigma_conv

    # Row 4 (theta_s_iw)
    Y[4, 2] = inv_R_a_iw
    Y[4, 3] = sigma_iw # Q_hc radiant to IW surface
    Y[4, 4] = -(inv_R_a_iw + inv_R_1_iw)
    Y[4, 5] = inv_R_1_iw

    # Row 5 (theta_m_iw)
    Y[5, 4] = inv_R_1_iw
    Y[5, 5] = -inv_R_1_iw - C_1_iw / tau

    return Y

def calc_q_tset(R_rest_aw, R_alpha_star_il, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                T_set, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                phi_conv, phi_aw, phi_iw, tau):
    """Calculate q matrix for 7R2C setpoint mode."""
    inv_R_rest_aw = 1.0 / R_rest_aw
    inv_R_a_il = 1.0 / R_alpha_star_il
    return calc_q_tset_optim(inv_R_rest_aw, inv_R_a_il, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                             T_set, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                             phi_conv, phi_aw, phi_iw, tau)


def calc_q_tset_optim(inv_R_r_aw, inv_R_a_il, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                      T_set, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                      phi_conv, phi_aw, phi_iw, tau):
    """Calculate q matrix for 7R2C setpoint mode in optimized way."""
    q = np.zeros(6, dtype=np.float64)

    # Row 0 (theta_m_aw)
    q[0] = -inv_R_r_aw * T_eq - C_1_aw * Tm_aw_prev / tau

    # Row 1 (theta_s_aw)
    q[1] = -phi_aw

    # Row 2 (theta_lu_star)
    q[2] = -inv_R_a_il * T_set

    # Row 3 (air node balance uses Q_hc unknown, location col=3)
    q[3] = (
        inv_R_a_il * T_set
        - phi_conv
        - Hve_inf * (T_out - T_set)
        - Hve_vent * (T_sup - T_set)
        + C_air * (T_set - Ta_prev) / tau
    )

    # Row 4 (theta_s_iw)
    q[4] = -phi_iw

    # Row 5 (theta_m_iw)
    q[5] = -C_1_iw * Tm_iw_prev / tau

    return q


def calc_y_phiset(R_1_aw, R_1_iw, R_alpha_star_il, R_alpha_star_aw, R_alpha_star_iw, R_rest_aw, C_1_aw, C_1_iw, C_air,
                  Hve_inf, Hve_vent, tau):
    """Calculate Y matrix for 7R2C setpoint mode."""
    # Precompute conductances (1/R) — speeds things up
    inv_R_1_aw = 1.0 / R_1_aw
    inv_R_1_iw = 1.0 / R_1_iw
    inv_R_a_il = 1.0 / R_alpha_star_il
    inv_R_a_aw = 1.0 / R_alpha_star_aw
    inv_R_a_iw = 1.0 / R_alpha_star_iw
    inv_R_r_aw = 1.0 / R_rest_aw
    Hve_inf = float(Hve_inf)
    Hve_vent = float(Hve_vent)

    return calc_y_phiset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                               C_1_aw, C_1_iw, C_air, Hve_inf, Hve_vent, tau)


def calc_y_phiset_optim(inv_R_1_aw, inv_R_1_iw, inv_R_a_il, inv_R_a_aw, inv_R_a_iw, inv_R_r_aw,
                        C_1_aw, C_1_iw, C_air, Hve_inf, Hve_vent, tau):
    """Calculate Y matrix for 7R2C setpoint mode in optimized way."""
    # Build Y (6x6)
    Y = np.zeros((6, 6), dtype=np.float64)

    # Row 0 (theta_m_aw)
    Y[0, 0] = -(inv_R_r_aw + inv_R_1_aw) - C_1_aw / tau
    Y[0, 1] = inv_R_1_aw

    # Row 1 (theta_s_aw)
    Y[1, 0] = inv_R_1_aw
    Y[1, 1] = -(inv_R_1_aw + inv_R_a_aw)
    Y[1, 2] = inv_R_a_aw

    # Row 2 (theta_lu_star)
    Y[2, 1] = inv_R_a_aw
    Y[2, 2] = -(inv_R_a_aw + inv_R_a_il + inv_R_a_iw)
    Y[2, 3] = inv_R_a_il
    Y[2, 4] = inv_R_a_iw

    # Row 3 (air node)
    Y[3, 2] = inv_R_a_il
    Y[3, 3] = -inv_R_a_il - Hve_inf - Hve_vent - C_air / tau

    # Row 4 (theta_s_iw)
    Y[4, 2] = inv_R_a_iw
    Y[4, 4] = -(inv_R_a_iw + inv_R_1_iw)
    Y[4, 5] = inv_R_1_iw

    # Row 5 (theta_m_iw)
    Y[5, 4] = inv_R_1_iw
    Y[5, 5] = -inv_R_1_iw - C_1_iw / tau

    return Y


def calc_q_phiset(R_rest_aw, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                  Q_hk_aw, Q_hk_iw, Q_hk_kon, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                  phi_conv, phi_aw, phi_iw, tau):
    inv_R_rest_aw = 1.0 / R_rest_aw
    return calc_q_phiset_optim(inv_R_rest_aw, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                               Q_hk_aw, Q_hk_iw, Q_hk_kon, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                               phi_conv, phi_aw, phi_iw, tau)


def calc_q_phiset_optim(inv_R_r_aw, Hve_inf, Hve_vent, C_1_aw, C_1_iw, C_air,
                        Q_hk_aw, Q_hk_iw, Q_hk_kon, T_eq, T_out, T_sup, Tm_aw_prev, Ta_prev, Tm_iw_prev,
                        phi_conv, phi_aw, phi_iw, tau):
    q = np.zeros(6, dtype=np.float64)

    # Row 0 (theta_m_aw)
    q[0] = -inv_R_r_aw * T_eq - C_1_aw * Tm_aw_prev / tau

    # Row 1 (theta_s_aw)
    q[1] = -(Q_hk_aw + phi_aw)

    # Row 2 (theta_lu_star)
    # q[2] stays 0

    # Row 3 (air node)
    q[3] = -Q_hk_kon - phi_conv - Hve_inf * T_out - Hve_vent * T_sup - C_air * Ta_prev / tau

    # Row 4 (theta_s_iw)
    q[4] = -(Q_hk_iw + phi_iw)

    # Row 5 (theta_m_iw)
    q[5] = -C_1_iw * Tm_iw_prev / tau

    return q
