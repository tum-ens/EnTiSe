import logging
from typing import Optional

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
    DEFAULT_ACTIVE_INTERNAL_GAINS,
    DEFAULT_ACTIVE_SOLAR_GAINS,
    DEFAULT_ACTIVE_VENTILATION,
    DEFAULT_FRAC_CONV_INTERNAL,
    DEFAULT_FRAC_RAD_MASS,
    DEFAULT_FRAC_RAD_SURFACE,
    DEFAULT_POWER_COOLING,
    DEFAULT_POWER_HEATING,
    DEFAULT_SIGMA_R5C1,
    DEFAULT_TEMP_INIT,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_VENTILATION,
)

logger = logging.getLogger(__name__)

# Module-level caches (per process)
_WEATHER_CACHE: dict[tuple, pd.DataFrame] = {}


class R5C1(Method):
    """5R1C (ISO 13790) HVAC demand and temperature simulation.

    This method simulates the heating and cooling demand of a building using the 5R1C model as defined in ISO 13790.
    It accounts for thermal capacitance, resistances, internal gains, solar gains, and ventilation effects.

    Attributes:
        types (list): List of method types.
        name (str): Name of the method.
        required_keys (list): List of required object keys.
        optional_keys (list): List of optional object keys.
        required_timeseries (list): List of required timeseries keys.
        optional_timeseries (list): List of optional timeseries keys.
        output_summary (dict): Summary of output metrics.
        output_timeseries (dict): Timeseries output metrics.
    """

    types = [Types.HVAC]
    name = "5R1C"
    required_keys = [
        O.H_TR_IS,
        O.H_TR_MS,
        O.H_TR_W,
        O.H_TR_EM,
        O.C_M,
        O.WEATHER,
    ]
    optional_keys = [
        O.POWER_HEATING,
        O.POWER_COOLING,
        O.ACTIVE_HEATING,
        O.ACTIVE_COOLING,
        O.ACTIVE_GAINS_INTERNAL,
        O.ACTIVE_GAINS_SOLAR,
        O.ACTIVE_VENTILATION,
        O.H_VE,
        O.TEMP_INIT,
        O.TEMP_MIN,
        O.TEMP_MAX,
        O.TEMP_SUPPLY,
        O.AREA,
        O.HEIGHT,
        O.SIGMA_5R1C_SURFACE,
        O.FRAC_CONV_INTERNAL,
        O.FRAC_RAD_SURFACE,
        O.FRAC_RAD_MASS,
        O.GAINS_INTERNAL,
        O.GAINS_INTERNAL_COL,
        O.GAINS_SOLAR,
        O.VENTILATION,
        O.VENTILATION_COL,
    ]
    required_timeseries = [O.WEATHER]
    optional_timeseries = [O.GAINS_INTERNAL, O.GAINS_SOLAR, O.VENTILATION]
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

    def generate(self, obj: dict = None, data: dict = None, ts_type: str = Types.HVAC, **kwargs) -> dict:
        obj, data = self._process_kwargs(obj, data, **kwargs)
        obj, data = self._get_input_data(obj, data, ts_type)

        prep = self._prepare_inputs(obj, data)
        index = prep["index"]
        dt_s = prep["dt_s"]

        # 3) Solve per-step (solver implementation to be added)
        temp_in, p_heat, p_cool = calculate_timeseries(prep)

        # 4) Outputs
        return self._format_output(
            temp_in.round(3), p_heat.round().astype(int), p_cool.round().astype(int), index, dt_s
        )

    # ---- Internals ----
    @staticmethod
    def _get_input_data(obj: dict, data: dict, method_type: str = Types.HVAC) -> tuple[dict, dict]:
        """Process and validate input data for HVAC calculation (R5C1).

        - Resolves method-prefixed keys with fallback to shared keys
        - Normalizes and caches weather (index = Columns.DATETIME)
        - Passes through optional timeseries references
        - Accepts either direct `O.H_VE` or auxiliary `O.VENTILATION` inputs
        - Includes 5R1C-specific split parameters
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
            O.ACTIVE_GAINS_INTERNAL: Method.get_with_method_backup(
                obj, O.ACTIVE_GAINS_INTERNAL, method_type, DEFAULT_ACTIVE_INTERNAL_GAINS
            ),
            O.ACTIVE_GAINS_SOLAR: Method.get_with_method_backup(
                obj, O.ACTIVE_GAINS_SOLAR, method_type, DEFAULT_ACTIVE_SOLAR_GAINS
            ),
            O.ACTIVE_VENTILATION: Method.get_with_method_backup(
                obj, O.ACTIVE_VENTILATION, method_type, DEFAULT_ACTIVE_VENTILATION
            ),
            # Power limits
            O.POWER_COOLING: Method.get_with_method_backup(obj, O.POWER_COOLING, method_type, DEFAULT_POWER_COOLING),
            O.POWER_HEATING: Method.get_with_method_backup(obj, O.POWER_HEATING, method_type, DEFAULT_POWER_HEATING),
            # Temperature setpoints
            O.TEMP_INIT: Method.get_with_method_backup(obj, O.TEMP_INIT, method_type, DEFAULT_TEMP_INIT),
            O.TEMP_MAX: Method.get_with_method_backup(obj, O.TEMP_MAX, method_type, DEFAULT_TEMP_MAX),
            O.TEMP_MIN: Method.get_with_method_backup(obj, O.TEMP_MIN, method_type, DEFAULT_TEMP_MIN),
            O.TEMP_SUPPLY: Method.get_with_method_backup(obj, O.TEMP_SUPPLY, method_type),
            # 5R1C RC parameters
            O.C_M: Method.get_with_method_backup(obj, O.C_M, method_type),
            O.H_TR_IS: Method.get_with_method_backup(obj, O.H_TR_IS, method_type),
            O.H_TR_MS: Method.get_with_method_backup(obj, O.H_TR_MS, method_type),
            O.H_TR_W: Method.get_with_method_backup(obj, O.H_TR_W, method_type),
            O.H_TR_EM: Method.get_with_method_backup(obj, O.H_TR_EM, method_type),
            O.H_VE: Method.get_with_method_backup(obj, O.H_VE, method_type),
            # Splits
            O.SIGMA_5R1C_SURFACE: Method.get_with_method_backup(
                obj, O.SIGMA_5R1C_SURFACE, method_type, DEFAULT_SIGMA_R5C1
            ),
            O.FRAC_CONV_INTERNAL: Method.get_with_method_backup(
                obj, O.FRAC_CONV_INTERNAL, method_type, DEFAULT_FRAC_CONV_INTERNAL
            ),
            O.FRAC_RAD_SURFACE: Method.get_with_method_backup(
                obj, O.FRAC_RAD_SURFACE, method_type, DEFAULT_FRAC_RAD_SURFACE
            ),
            O.FRAC_RAD_MASS: Method.get_with_method_backup(obj, O.FRAC_RAD_MASS, method_type, DEFAULT_FRAC_RAD_MASS),
            # Internal gains
            O.GAINS_INTERNAL: Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type),
            O.GAINS_INTERNAL_COL: Method.get_with_method_backup(obj, O.GAINS_INTERNAL_COL, method_type),
            # Ventilation (back-up if H_ve not provided)
            O.VENTILATION: Method.get_with_method_backup(obj, O.VENTILATION, method_type, DEFAULT_VENTILATION),
            O.VENTILATION_COL: Method.get_with_method_backup(obj, O.VENTILATION_COL, method_type),
        }

        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather = Method.get_with_backup(data, weather_key)

        windows_key = Method.get_with_method_backup(obj, O.WINDOWS, method_type)
        windows = Method.get_with_backup(data, windows_key) if isinstance(windows_key, str) else None

        # Optional TS references
        internal_key = Method.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type)
        internal_gains = Method.get_with_backup(data, internal_key) if isinstance(internal_key, str) else None

        solar_key = Method.get_with_method_backup(obj, O.GAINS_SOLAR, method_type)
        solar_gains = Method.get_with_backup(data, solar_key) if isinstance(solar_key, str) else None

        ventilation_key = Method.get_with_method_backup(obj, O.VENTILATION, method_type)
        ventilation = Method.get_with_backup(data, ventilation_key) if isinstance(ventilation_key, str) else None

        data_out = {
            O.WEATHER: weather,
            O.WINDOWS: windows,
            internal_key: internal_gains,
            solar_key: solar_gains,
            ventilation_key: ventilation,
        }

        # Clean up Nones
        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        data_out = {k: v for k, v in data_out.items() if v is not None}

        # Weather normalization and caching
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

    def _prepare_inputs(self, obj: dict, data: dict) -> dict:
        """Prepare a solver-ready bundle for R5C1.

        Returns a dict with:
          - index, dt_s, weather
          - g_int_series, g_sol_series (W)
          - Hve_series (W/K), preferring O.H_VE if provided
          - controls: dict (setpoints, caps, activation flags)
          - params: dict (Cm, Htr_is, Htr_ms, Htr_w, Htr_em)
          - splits: dict (sigma, f_conv, f_rad_surf, f_rad_mass)
        """
        # Weather and timestep
        weather = data[O.WEATHER]
        index = weather.index
        idx = weather[C.DATETIME].values.astype("datetime64[ns]")
        dt_s = float((idx[1] - idx[0]) / np.timedelta64(1, "s"))

        # Gains (optional → auxiliaries depending on active flags)
        g_int_df = data.get(O.GAINS_INTERNAL)
        if g_int_df is None and obj.get(O.ACTIVE_GAINS_INTERNAL, DEFAULT_ACTIVE_INTERNAL_GAINS):
            g_int_df = InternalGains().generate(obj, data)
        if g_int_df is None:
            g_int_df = pd.DataFrame(0.0, index=index, columns=["g_int"])  # fallback

        g_sol_df = data.get(O.GAINS_SOLAR)
        if g_sol_df is None and obj.get(O.ACTIVE_GAINS_SOLAR, DEFAULT_ACTIVE_SOLAR_GAINS):
            g_sol_df = SolarGains().generate(obj, data)
        if g_sol_df is None:
            g_sol_df = pd.DataFrame(0.0, index=index, columns=["g_sol"])  # fallback

        g_int_series = g_int_df.squeeze()
        g_sol_series = g_sol_df.squeeze()

        # Ventilation normalization with precedence for H_ve
        Hve_series = self._resolve_ventilation(obj, data, weather)

        # Optional split: infiltration vs mechanical ventilation.
        # Default behavior: all in mechanical (vent) term → preserves current results.
        Hve_inf_series = pd.Series(0.0, index=index, name=f"{O.VENTILATION}_inf")
        Hve_vent_series = Hve_series.copy()
        Hve_vent_series.name = f"{O.VENTILATION}_vent"

        volume = float(obj.get(O.AREA, Const.DEFAULT_AREA.value)) * float(obj.get(O.HEIGHT, Const.DEFAULT_HEIGHT.value))
        rho_air = 1.2  # kg/m3
        cp_air = 1000.0  # J/kgK
        capacity_air = volume * rho_air * cp_air

        # Controls
        controls = dict(
            T_init=float(obj.get(O.TEMP_INIT, DEFAULT_TEMP_INIT)),
            T_min=float(obj.get(O.TEMP_MIN, DEFAULT_TEMP_MIN)),
            T_max=float(obj.get(O.TEMP_MAX, DEFAULT_TEMP_MAX)),
            P_h_max=float(obj.get(O.POWER_HEATING, DEFAULT_POWER_HEATING)),
            P_c_max=float(obj.get(O.POWER_COOLING, DEFAULT_POWER_COOLING)),
            on_h=bool(obj.get(O.ACTIVE_HEATING, DEFAULT_ACTIVE_HEATING)),
            on_c=bool(obj.get(O.ACTIVE_COOLING, DEFAULT_ACTIVE_COOLING)),
            T_sup=float(obj.get(O.TEMP_SUPPLY, np.nan)) if obj.get(O.TEMP_SUPPLY) is not None else None,
        )

        # 5R1C parameters
        params = dict(
            Cm=float(obj[O.C_M]),
            Htr_is=float(obj[O.H_TR_IS]),
            Htr_ms=float(obj[O.H_TR_MS]),
            Htr_w=float(obj[O.H_TR_W]),
            Htr_em=float(obj[O.H_TR_EM]),
            capacity_air=capacity_air,
        )

        # Splits
        sigma_surface = float(obj.get(O.SIGMA_5R1C_SURFACE, DEFAULT_SIGMA_R5C1))
        sigma = (sigma_surface, 1.0 - sigma_surface)  # (radiant to surfaces, convective to air)

        f_conv = float(obj.get(O.FRAC_CONV_INTERNAL, DEFAULT_FRAC_CONV_INTERNAL))
        f_rad_surf = float(obj.get(O.FRAC_RAD_SURFACE, DEFAULT_FRAC_RAD_SURFACE))
        f_rad_mass = float(obj.get(O.FRAC_RAD_MASS, DEFAULT_FRAC_RAD_MASS))
        s = max(f_rad_surf + f_rad_mass, 1e-9)
        f_rad_surf /= s
        f_rad_mass /= s

        # Optional ISO-13790 gain split parameters (area of mass surface and total area)
        A_m = obj.get("A_m")
        A_tot = obj.get("A_tot")

        splits = dict(
            sigma=sigma,
            f_conv=f_conv,
            f_rad_surf=f_rad_surf,
            f_rad_mass=f_rad_mass,
        )
        if A_m is not None and A_tot is not None:
            splits["A_m"] = float(A_m)
            splits["A_tot"] = float(A_tot)

        return dict(
            index=index,
            dt_s=dt_s,
            weather=weather,
            g_int_series=g_int_series,
            g_sol_series=g_sol_series,
            Hve_series=Hve_series,
            Hve_inf_series=Hve_inf_series,
            Hve_vent_series=Hve_vent_series,
            controls=controls,
            params=params,
            splits=splits,
        )

    def _resolve_ventilation(self, obj: dict, data: dict, weather: pd.DataFrame) -> pd.Series:
        index = weather.index
        # 1) Prefer H_ve if present
        if O.H_VE in obj and obj[O.H_VE] is not None:
            val = obj[O.H_VE]
            # If a pandas Series: align/validate
            if isinstance(val, pd.Series):
                if not val.index.equals(index):
                    raise ValueError("H_ve series index does not match weather index.")
                logger.info("Using H_ve series as ventilation conductance (overrides auxiliary ventilation)")
                return val.astype(float)
            # Else: scalar → expand
            try:
                fval = float(val)
                logger.info("Using H_ve scalar as ventilation conductance (overrides auxiliary ventilation)")
                return pd.Series(np.full(len(index), fval, dtype=float), index=index, name=O.VENTILATION)
            except Exception as err:
                raise ValueError(f"H_ve must be a float or a pandas Series, got {type(val)}") from err

        # 2) If a ventilation DataFrame already exists in data, use it
        ven_df = data.get(O.VENTILATION)
        if isinstance(ven_df, pd.DataFrame) and O.VENTILATION in ven_df:
            return ven_df[O.VENTILATION].astype(float)

        # 3) If obj[O.VENTILATION] is numeric, expand constant
        if O.VENTILATION in obj:
            try:
                fval = float(obj[O.VENTILATION])
                return pd.Series(np.full(len(index), fval, dtype=float), index=index, name=O.VENTILATION)
            except Exception:
                pass  # could be a string key for auxiliary

        # 4) If active, try auxiliary to generate it (supports strings and columns)
        if obj.get(O.ACTIVE_VENTILATION, DEFAULT_ACTIVE_VENTILATION):
            ven_df = Ventilation().generate(obj, {**data, O.WEATHER: weather})
            if not isinstance(ven_df, pd.DataFrame) or O.VENTILATION not in ven_df:
                raise ValueError("Ventilation auxiliary did not return a DataFrame with VENTILATION column")
            return ven_df[O.VENTILATION].astype(float)

        # 5) Fallback default
        logger.info("No H_ve or ventilation provided; using DEFAULT_VENTILATION")
        return pd.Series(np.full(len(index), DEFAULT_VENTILATION, dtype=float), index=index, name=O.VENTILATION)

    @staticmethod
    def _format_output(temp_in, p_heat, p_cool, index, dt_s) -> dict:
        E_h_Wh = float(np.trapezoid(p_heat.values, dx=dt_s) / 3600.0)
        E_c_Wh = float(np.trapezoid(p_cool.values, dx=dt_s) / 3600.0)
        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": E_h_Wh,
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": float(max(p_heat.max(), 0.0)),
            f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": E_c_Wh,
            f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": float(max(p_cool.max(), 0.0)),
        }

        df = pd.DataFrame(
            {
                C.TEMP_IN: temp_in,
                f"{Types.HEATING}{SEP}{C.LOAD}[W]": p_heat,
                f"{Types.COOLING}{SEP}{C.LOAD}[W]": p_cool,
            },
            index=index,
        )
        df.index.name = C.DATETIME

        return {"summary": summary, "timeseries": df}


def calculate_timeseries(prep: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run the 5R1C simulation over the full index using the prepared bundle."""
    index = prep["index"]
    dt_s: float = prep["dt_s"]
    weather: pd.DataFrame = prep["weather"]
    g_int_series: pd.Series = prep["g_int_series"]
    g_sol_series: pd.Series = prep["g_sol_series"]
    Hve_series: pd.Series = prep["Hve_series"]
    Hve_inf_series: pd.Series = prep.get("Hve_inf_series")
    Hve_vent_series: pd.Series = prep.get("Hve_vent_series")

    if Hve_inf_series is None or Hve_vent_series is None:
        # Backward-compatible fallback: all ventilation treated as mechanical at T_sup
        Hve_inf_series = pd.Series(0.0, index=Hve_series.index, name=f"{O.VENTILATION}_inf")
        Hve_vent_series = Hve_series.copy()
        Hve_vent_series.name = f"{O.VENTILATION}_vent"

    controls = prep["controls"]
    T_init: float = controls["T_init"]
    T_min: float = controls["T_min"]
    T_max: float = controls["T_max"]
    P_h_max: float = controls["P_h_max"]
    P_c_max: float = controls["P_c_max"]
    on_h: bool = controls["on_h"]
    on_c: bool = controls["on_c"]
    T_sup_opt = controls.get("T_sup")

    params = prep["params"]
    Cm = float(params["Cm"])
    Htr_is = float(params["Htr_is"])
    Htr_ms = float(params["Htr_ms"])
    Htr_w = float(params["Htr_w"])
    Htr_em = float(params["Htr_em"])
    capacity_air = float(params["capacity_air"])

    splits = prep["splits"]
    A_m = splits.get("A_m")
    A_tot = splits.get("A_tot")
    sigma_surface, sigma_conv = _normalize_sigma(tuple(splits["sigma"]))
    f_conv = float(splits["f_conv"])
    f_rad_surf = float(splits["f_rad_surf"])
    f_rad_mass = float(splits["f_rad_mass"])

    Ta = float(T_init)
    Tm = float(T_init)

    t_in_values: list[float] = []
    qh_values: list[float] = []
    qc_values: list[float] = []

    for ts in index:
        T_e = float(weather.at[ts, C.TEMP_AIR])
        T_sup = (
            T_e if (T_sup_opt is None or (isinstance(T_sup_opt, float) and np.isnan(T_sup_opt))) else float(T_sup_opt)
        )

        Hve_inf = float(Hve_inf_series.at[ts])
        Hve_vent = float(Hve_vent_series.at[ts])

        g_int = float(g_int_series.at[ts])
        g_sol = float(g_sol_series.at[ts])
        phi_ia, phi_st, phi_m = _split_gains_5r1c(
            g_int,
            g_sol,
            f_conv,
            f_rad_surf,
            f_rad_mass,
            Htr_w=Htr_w,
            A_m=A_m,
            A_tot=A_tot,
        )

        # 1) Free-float prediction (no HVAC)
        sol_free = _solve_step_phiset(
            Cm=Cm,
            Htr_is=Htr_is,
            Htr_ms=Htr_ms,
            Htr_w=Htr_w,
            Htr_em=Htr_em,
            Hve_inf=Hve_inf,
            Hve_vent=Hve_vent,
            capacity_air=capacity_air,
            T_e=T_e,
            T_sup=T_sup,
            phi_ia=phi_ia,
            phi_st=phi_st,
            phi_m=phi_m,
            sigma_surface=sigma_surface,
            sigma_conv=sigma_conv,
            Q_hc=0.0,
            Tm_prev=Tm,
            Ta_prev=Ta,
            dt_s=dt_s,
        )
        Ta_free = float(sol_free[1])
        Tm_free = float(sol_free[3])

        Q_applied = 0.0

        # 2) Decide whether we need heating or cooling based on free-float air temperature
        if Ta_free < T_min and on_h:
            # Need heating to keep at lower bound
            sol_tset = _solve_step_tset(
                Cm=Cm,
                Htr_is=Htr_is,
                Htr_ms=Htr_ms,
                Htr_w=Htr_w,
                Htr_em=Htr_em,
                Hve_inf=Hve_inf,
                Hve_vent=Hve_vent,
                capacity_air=capacity_air,
                T_e=T_e,
                T_sup=T_sup,
                phi_ia=phi_ia,
                phi_st=phi_st,
                phi_m=phi_m,
                sigma_surface=sigma_surface,
                sigma_conv=sigma_conv,
                T_set=T_min,
                Tm_prev=Tm,
                Ta_prev=Ta,
                dt_s=dt_s,
            )
            Q_req = float(sol_tset[0])
            Q_applied, clamped = _clamp_hvac(Q_req, on_h, on_c, P_h_max, P_c_max)
            if clamped:
                # Recompute temperatures with capped power
                sol_phi = _solve_step_phiset(
                    Cm=Cm,
                    Htr_is=Htr_is,
                    Htr_ms=Htr_ms,
                    Htr_w=Htr_w,
                    Htr_em=Htr_em,
                    Hve_inf=Hve_inf,
                    Hve_vent=Hve_vent,
                    capacity_air=capacity_air,
                    T_e=T_e,
                    T_sup=T_sup,
                    phi_ia=phi_ia,
                    phi_st=phi_st,
                    phi_m=phi_m,
                    sigma_surface=sigma_surface,
                    sigma_conv=sigma_conv,
                    Q_hc=Q_applied,
                    Tm_prev=Tm,
                    Ta_prev=Ta,
                    dt_s=dt_s,
                )
                Ta = float(sol_phi[1])
                Tm = float(sol_phi[3])
            else:
                # Ideal case: we hit the setpoint exactly
                Ta = float(sol_tset[1])  # T_min
                Tm = float(sol_tset[3])

        elif Ta_free > T_max and on_c:
            # Need cooling to keep at upper bound
            sol_tset = _solve_step_tset(
                Cm=Cm,
                Htr_is=Htr_is,
                Htr_ms=Htr_ms,
                Htr_w=Htr_w,
                Htr_em=Htr_em,
                Hve_inf=Hve_inf,
                Hve_vent=Hve_vent,
                capacity_air=capacity_air,
                T_e=T_e,
                T_sup=T_sup,
                phi_ia=phi_ia,
                phi_st=phi_st,
                phi_m=phi_m,
                sigma_surface=sigma_surface,
                sigma_conv=sigma_conv,
                T_set=T_max,
                Tm_prev=Tm,
                Ta_prev=Ta,
                dt_s=dt_s,
            )
            Q_req = float(sol_tset[0])
            Q_applied, clamped = _clamp_hvac(Q_req, on_h, on_c, P_h_max, P_c_max)
            if clamped:
                sol_phi = _solve_step_phiset(
                    Cm=Cm,
                    Htr_is=Htr_is,
                    Htr_ms=Htr_ms,
                    Htr_w=Htr_w,
                    Htr_em=Htr_em,
                    Hve_inf=Hve_inf,
                    Hve_vent=Hve_vent,
                    capacity_air=capacity_air,
                    T_e=T_e,
                    T_sup=T_sup,
                    phi_ia=phi_ia,
                    phi_st=phi_st,
                    phi_m=phi_m,
                    sigma_surface=sigma_surface,
                    sigma_conv=sigma_conv,
                    Q_hc=Q_applied,
                    Tm_prev=Tm,
                    Ta_prev=Ta,
                    dt_s=dt_s,
                )
                Ta = float(sol_phi[1])
                Tm = float(sol_phi[3])
            else:
                Ta = float(sol_tset[1])  # T_max
                Tm = float(sol_tset[3])

        else:
            # Free-float stays within band or HVAC off
            Ta = Ta_free
            Tm = Tm_free

        t_in_values.append(Ta)
        qh_values.append(max(Q_applied, 0.0))
        qc_values.append(max(-Q_applied, 0.0))

    T_in = pd.Series(t_in_values, index=index, dtype=float)
    Q_heat = pd.Series(qh_values, index=index, dtype=float)
    Q_cool = pd.Series(qc_values, index=index, dtype=float)
    return T_in, Q_heat, Q_cool


def _split_gains_5r1c(
    g_int: float,
    g_sol: float,
    f_conv: float,
    f_rad_surf: float,
    f_rad_mass: float,
    Htr_w: Optional[float] = None,
    A_m: Optional[float] = None,
    A_tot: Optional[float] = None,
) -> tuple[float, float, float]:
    """Split gains into (convective to air, radiant to surfaces, radiant to mass).

    If Htr_w, A_m and A_tot are provided, use ISO 13790-like split
    (Eureca-style): convective internal gains go to air, radiant + solar
    are distributed between mass and surfaces, with a window-related term.

    Otherwise, fall back to user-defined fractions.
    """
    g_int = float(g_int)
    g_sol = float(g_sol)

    # ISO-13790 / Eureca-style split when enough information is provided
    if Htr_w is not None and A_m is not None and A_tot is not None and A_tot > 0.0:
        A_m = float(A_m)
        A_tot = float(A_tot)
        Htr_w = float(Htr_w)

        # Convective part of internal gains only
        g_int_conv = f_conv * g_int
        g_int_rad = (1.0 - f_conv) * g_int

        phi_ia = g_int_conv

        phi_rad_plus_sol = g_int_rad + g_sol  # radiant + solar

        phi_m = (A_m / A_tot) * phi_rad_plus_sol
        phi_st = (1.0 - A_m / A_tot - Htr_w / (9.1 * A_tot)) * phi_rad_plus_sol

        return float(phi_ia), float(phi_st), float(phi_m)

    # Default: user-defined fractional split
    g_tot = float(g_int + g_sol)
    phi_ia = float(f_conv * g_tot)
    phi_rad = float((1.0 - f_conv) * g_tot)
    # `f_rad_surf + f_rad_mass` already normalized in `_prepare_inputs`
    phi_st = float(f_rad_surf * phi_rad)
    phi_m = float(f_rad_mass * phi_rad)
    return phi_ia, phi_st, phi_m


def _normalize_sigma(sigma: tuple[float, float]) -> tuple[float, float]:
    """Ensure sigma components sum to 1. Returns (sigma_surface, sigma_conv)."""
    s = float(sigma[0] + sigma[1])
    if s <= 0.0:
        return 0.5, 0.5
    return float(sigma[0] / s), float(sigma[1] / s)


def _clamp_hvac(Q_req: float, on_h: bool, on_c: bool, P_h_max: float, P_c_max: float) -> tuple[float, bool]:
    """Apply heating/cooling caps and activation flags. Positive = heating, negative = cooling.

    Returns:
        (Q_applied, clamped)
    """
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


def _solve_step_tset(
    *,
    Cm: float,
    Htr_is: float,
    Htr_ms: float,
    Htr_w: float,
    Htr_em: float,
    Hve_inf: float,
    Hve_vent: float,
    capacity_air: float,
    T_e: float,
    T_sup: float,
    phi_ia: float,
    phi_st: float,
    phi_m: float,
    sigma_surface: float,
    sigma_conv: float,
    T_set: float,
    Tm_prev: float,
    Ta_prev: float,
    dt_s: float,
) -> np.ndarray:
    """One-step 5R1C solve in setpoint mode (solve for required HVAC power).

    Returns array: [Q_hc_req, T_air=T_set, T_s, T_m].
    """
    # Unknowns: [Q_hc, T_s, T_m]
    Y = np.zeros((3, 3), dtype=float)
    q = np.zeros(3, dtype=float)

    Y[0, 0] = sigma_conv
    Y[0, 1] = Htr_is

    Y[1, 0] = sigma_surface
    Y[1, 1] = -(Htr_is + Htr_w + Htr_ms)
    Y[1, 2] = Htr_ms

    Y[2, 1] = Htr_ms
    Y[2, 2] = -(Cm / dt_s + Htr_em + Htr_ms)

    q[0] = (
        Hve_inf * (T_set - T_e)
        + Hve_vent * (T_set - T_sup)
        - phi_ia
        + Htr_is * T_set
        + capacity_air * (T_set - Ta_prev) / dt_s
    )
    q[1] = -Htr_is * T_set - phi_st - Htr_w * T_e
    q[2] = -Htr_em * T_e - phi_m - Cm * Tm_prev / dt_s

    x = np.linalg.solve(Y, q)
    return np.array([x[0], T_set, x[1], x[2]], dtype=float)


def _solve_step_phiset(
    *,
    Cm: float,
    Htr_is: float,
    Htr_ms: float,
    Htr_w: float,
    Htr_em: float,
    Hve_inf: float,
    Hve_vent: float,
    capacity_air: float,
    T_e: float,
    T_sup: float,
    phi_ia: float,
    phi_st: float,
    phi_m: float,
    sigma_surface: float,
    sigma_conv: float,
    Q_hc: float,
    Tm_prev: float,
    Ta_prev: float,
    dt_s: float,
) -> np.ndarray:
    """One-step 5R1C solve in applied-load mode (compute resultant temperatures).

    Returns array: [Q_hc, T_air, T_s, T_m].
    """
    # Unknowns: [T_air, T_s, T_m]
    Y = np.zeros((3, 3), dtype=float)
    q = np.zeros(3, dtype=float)

    Hve_tot = Hve_inf + Hve_vent

    Y[0, 0] = -(Htr_is + Hve_tot) - capacity_air / dt_s
    Y[0, 1] = Htr_is

    Y[1, 0] = Htr_is
    Y[1, 1] = -(Htr_is + Htr_w + Htr_ms)
    Y[1, 2] = Htr_ms

    Y[2, 1] = Htr_ms
    Y[2, 2] = -(Cm / dt_s + Htr_em + Htr_ms)

    # Ventilation split in source term
    q[0] = -Q_hc * sigma_conv - (Hve_inf * T_e + Hve_vent * T_sup) - phi_ia - capacity_air * Ta_prev / dt_s
    q[1] = -Q_hc * sigma_surface - phi_st - Htr_w * T_e
    q[2] = -Htr_em * T_e - phi_m - Cm * Tm_prev / dt_s

    y = np.linalg.solve(Y, q)
    return np.array([Q_hc, y[0], y[1], y[2]], dtype=float)
