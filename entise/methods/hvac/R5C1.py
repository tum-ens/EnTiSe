import logging
from typing import Optional

import numpy as np
import pandas as pd

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Constants as Const
from entise.constants import Objects as O
from entise.core.base import Method
from entise.core.utils import resolve_ts_or_scalar
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.strategies import SolarGainsISO13790
from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries
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
    DEFAULT_SIGMA_5R1C,
    DEFAULT_TEMP_INIT,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_VENTILATION,
    DEFAULT_VENTILATION_SPLIT,
)

logger = logging.getLogger(__name__)

# Module-level caches (per process)
_WEATHER_CACHE: dict[tuple, pd.DataFrame] = {}


class R5C1(Method):
    """5R1C HVAC model aligned with ISO 13790's simplified dynamic method.

    Purpose and scope:
    - Captures key heat transfer paths between indoor air, internal surfaces,
      thermal mass, and exterior using five resistances and one aggregated
      capacitance (building mass). Better represents radiant/convective splits
      and envelope interactions than 1R1C, while remaining efficient for
      large‑scale simulations.

    Conceptual structure:
    - Capacitance C_m (building thermal mass) exchanges with internal surfaces
      via H_tr,ms and with indoor air via H_tr,is; windows and opaque elements
      couple to exterior via H_tr,w and H_tr,em. Optional sky correction via H_tr,op,sky.
    - Internal and solar gains are split into radiant/convective parts and routed
      to air, surfaces, and mass using σ parameters.
    - Ventilation losses are handled via H_ve (scalar or timeseries).

    Notes:
    - Implements the ISO 13790 simplified dynamic method assumptions (lumped mass
      and linear heat transfer). Parameter mapping follows standard notation.
    - For even richer transient behavior and phase shifts, consider a 7R2C model
      (see VDI 6007).

    Reference:
    - ISO 13790: Energy performance of buildings — Calculation of energy use for
      space heating and cooling (simplified dynamic method).
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
        # Controls
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
        O.TEMP_SUPPLY,
        O.AREA,
        O.HEIGHT,
        O.AREA_M,
        O.AREA_TOT,
        # Ventilation
        O.H_VE,
        # Sky facing surface
        O.H_TR_OP_SKY,
        # 5R1C splits
        O.SIGMA_SURFACE,
        # Gains splits
        O.FRAC_CONV_INTERNAL,
        O.FRAC_RAD_SURFACE,
        O.FRAC_RAD_MASS,
        # Timeseries
        O.WINDOWS,
        O.GAINS_INTERNAL,
        O.GAINS_SOLAR,
        O.VENTILATION,
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
        self, obj: dict = None, data: dict = None, results: dict = None, ts_type: str = Types.HVAC, **kwargs
    ) -> dict:
        obj, data = self._process_kwargs(obj, data, **kwargs)
        obj, data = self._get_input_data(obj, data, ts_type)
        data = self._prepare_inputs(obj, data)

        temp_in, p_heat, p_cool = calculate_timeseries_5r1c(**data)
        meta = data["meta"]
        return self._format_output(
            temp_in.round(3), p_heat.round().astype(int), p_cool.round().astype(int), meta["index"], meta["dt_s"]
        )

    # ---- Internals ----
    def _get_input_data(self, obj: dict, data: dict, method_type: str = Types.HVAC) -> tuple[dict, dict]:
        """Process and validate input data for HVAC calculation (R5C1).

        - Resolves method-prefixed keys with fallback to shared keys
        - Normalizes and caches weather (index = Columns.DATETIME)
        - Passes through optional timeseries references
        - Accepts either direct `O.H_VE` or auxiliary `O.VENTILATION` inputs
        - Includes 5R1C-specific split parameters
        """
        obj_out = self._get_relevant_objects(obj, method_type)
        data_out = self._prepare_data_tables(obj, data, method_type)
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
        data_out["meta"] = meta

        # Gains (generate if active and missing)
        data_out["series"] = {}
        gains = self._compute_gains(obj, data)
        data_out["series"]["gains"] = gains

        # Ventilation → split into mechanical & infiltration
        ven_df = VentilationTimeSeries().generate(obj, data).squeeze()  # Calls ISO 13709 conform norm directly
        vent_split = resolve_ts_or_scalar(obj, data, O.VENTILATION_SPLIT, index, default=DEFAULT_VENTILATION_SPLIT)
        Hve_vent_series = ven_df * vent_split
        Hve_inf_series = ven_df * (1.0 - vent_split)
        ventilation = pd.concat([Hve_vent_series, Hve_inf_series], axis=1, keys=["Hve_vent", "Hve_inf"])
        data_out["series"]["ventilation"] = ventilation

        # Controls
        temp_sup = obj.get(O.TEMP_SUPPLY, None)
        temp_sup = (
            resolve_ts_or_scalar(obj, data, O.TEMP_SUPPLY, index) if temp_sup is not None else weather[C.TEMP_AIR]
        )
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
        data_out["controls"] = controls

        # Parameters
        volume = float(obj.get(O.AREA, Const.DEFAULT_AREA.value)) * float(obj.get(O.HEIGHT, Const.DEFAULT_HEIGHT.value))
        rho_air = 1.2  # kg/m3
        cp_air = 1000.0  # J/kgK
        capacity_air = volume * rho_air * cp_air
        params = {
            O.TEMP_INIT: float(obj.get(O.TEMP_INIT, DEFAULT_TEMP_INIT)),
            O.C_M: float(obj[O.C_M]),
            O.H_TR_IS: float(obj[O.H_TR_IS]),
            O.H_TR_MS: float(obj[O.H_TR_MS]),
            O.H_TR_W: float(obj[O.H_TR_W]),
            O.H_TR_EM: float(obj[O.H_TR_EM]),
            O.CAPACITANCE_AIR: float(capacity_air),
        }
        data_out["params"] = params

        # Splits
        sigma_surface = float(obj.get(O.SIGMA_SURFACE, DEFAULT_SIGMA_5R1C))
        f_conv = float(obj.get(O.FRAC_CONV_INTERNAL, DEFAULT_FRAC_CONV_INTERNAL))
        f_rad_surf = float(obj.get(O.FRAC_RAD_SURFACE, DEFAULT_FRAC_RAD_SURFACE))
        f_rad_mass = float(obj.get(O.FRAC_RAD_MASS, DEFAULT_FRAC_RAD_MASS))
        s = max(f_rad_surf + f_rad_mass, 1e-9)
        f_rad_surf /= s
        f_rad_mass /= s
        splits = {
            O.SIGMA_SURFACE: sigma_surface,  # radiant to surfaces
            "sigma_conv": 1.0 - sigma_surface,  # convective to air
            O.FRAC_CONV_INTERNAL: f_conv,
            O.FRAC_RAD_SURFACE: f_rad_surf,
            O.FRAC_RAD_MASS: f_rad_mass,
            O.AREA_M: obj.get(O.AREA_M),
            O.AREA_TOT: obj.get(O.AREA_TOT),
        }
        data_out["splits"] = splits

        return data_out

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

    def _get_relevant_objects(self, obj: dict, method_type: str = Types.HVAC) -> dict:
        """Get relevant objects for the given method type, including method-prefixed keys."""
        # TODO: Change this so that it takes the data from self.objects and make this for loops.
        #  This probably also means changing the lists to dicts and having default values as the value
        obj_out = {
            O.ID: self.get_with_backup(obj, O.ID),
            # Geometry
            O.AREA: self.get_with_method_backup(obj, O.AREA, method_type, Const.DEFAULT_AREA.value),
            O.AREA_M: self.get_with_method_backup(obj, O.AREA_M, method_type),
            O.AREA_TOT: self.get_with_method_backup(obj, O.AREA_TOT, method_type),
            O.HEIGHT: self.get_with_method_backup(obj, O.HEIGHT, method_type, Const.DEFAULT_HEIGHT.value),
            O.LAT: self.get_with_method_backup(obj, O.LAT, method_type),
            O.LON: self.get_with_method_backup(obj, O.LON, method_type),
            # Controls
            O.ACTIVE_COOLING: self.get_with_method_backup(obj, O.ACTIVE_COOLING, method_type, DEFAULT_ACTIVE_COOLING),
            O.ACTIVE_HEATING: self.get_with_method_backup(obj, O.ACTIVE_HEATING, method_type, DEFAULT_ACTIVE_HEATING),
            O.ACTIVE_GAINS_INTERNAL: self.get_with_method_backup(
                obj, O.ACTIVE_GAINS_INTERNAL, method_type, DEFAULT_ACTIVE_INTERNAL_GAINS
            ),
            O.ACTIVE_GAINS_SOLAR: self.get_with_method_backup(
                obj, O.ACTIVE_GAINS_SOLAR, method_type, DEFAULT_ACTIVE_SOLAR_GAINS
            ),
            O.ACTIVE_VENTILATION: self.get_with_method_backup(
                obj, O.ACTIVE_VENTILATION, method_type, DEFAULT_ACTIVE_VENTILATION
            ),
            # Power limits
            O.POWER_COOLING: self.get_with_method_backup(obj, O.POWER_COOLING, method_type, DEFAULT_POWER_COOLING),
            O.POWER_HEATING: self.get_with_method_backup(obj, O.POWER_HEATING, method_type, DEFAULT_POWER_HEATING),
            # Temperature setpoints
            O.TEMP_INIT: self.get_with_method_backup(obj, O.TEMP_INIT, method_type, DEFAULT_TEMP_INIT),
            O.TEMP_MAX: self.get_with_method_backup(obj, O.TEMP_MAX, method_type, DEFAULT_TEMP_MAX),
            O.TEMP_MIN: self.get_with_method_backup(obj, O.TEMP_MIN, method_type, DEFAULT_TEMP_MIN),
            O.TEMP_SUPPLY: self.get_with_method_backup(obj, O.TEMP_SUPPLY, method_type),
            # 5R1C RC parameters
            O.C_M: self.get_with_method_backup(obj, O.C_M, method_type),
            O.H_TR_IS: self.get_with_method_backup(obj, O.H_TR_IS, method_type),
            O.H_TR_MS: self.get_with_method_backup(obj, O.H_TR_MS, method_type),
            O.H_TR_W: self.get_with_method_backup(obj, O.H_TR_W, method_type),
            O.H_TR_EM: self.get_with_method_backup(obj, O.H_TR_EM, method_type),
            O.H_VE: self.get_with_method_backup(obj, O.H_VE, method_type),
            O.H_TR_OP_SKY: self.get_with_method_backup(obj, O.H_TR_OP_SKY, method_type),
            # Splits
            O.SIGMA_SURFACE: self.get_with_method_backup(obj, O.SIGMA_SURFACE, method_type, DEFAULT_SIGMA_5R1C),
            O.FRAC_CONV_INTERNAL: self.get_with_method_backup(
                obj, O.FRAC_CONV_INTERNAL, method_type, DEFAULT_FRAC_CONV_INTERNAL
            ),
            O.FRAC_RAD_SURFACE: self.get_with_method_backup(
                obj, O.FRAC_RAD_SURFACE, method_type, DEFAULT_FRAC_RAD_SURFACE
            ),
            O.FRAC_RAD_MASS: self.get_with_method_backup(obj, O.FRAC_RAD_MASS, method_type, DEFAULT_FRAC_RAD_MASS),
            # Internal gains
            O.GAINS_INTERNAL: self.get_with_method_backup(obj, O.GAINS_INTERNAL, method_type),
            O.GAINS_INTERNAL_COL: self.get_with_method_backup(obj, O.GAINS_INTERNAL_COL, method_type),
            # Ventilation (back-up if H_ve not provided)
            O.VENTILATION: self.get_with_method_backup(obj, O.VENTILATION, method_type, DEFAULT_VENTILATION),
            O.VENTILATION_COL: self.get_with_method_backup(obj, O.VENTILATION_COL, method_type),
        }
        # Possible future impelementation. Also see 7R2C.py
        # relevant = {}
        # prefix = f"{method_type}{SEP}"
        # for key in obj:
        #     if key.startswith(prefix):
        #         relevant_key = key[len(prefix) :]
        #         relevant[relevant_key] = obj[key]
        #     elif key not in self.required_keys + self.optional_keys:
        #         continue  # skip unrelated keys
        #     else:
        #         relevant[key] = obj[key]
        return {k: v for k, v in obj_out.items() if v is not None}

    def _prepare_data_tables(self, obj: dict, data: dict, method_type: str = Types.HVAC) -> dict:
        data_out = {}

        for key in self.required_data:
            ts_key = self.get_with_method_backup(obj, key, method_type, key)
            ts_data = self.get_with_backup(data, ts_key)
            if ts_data is None:
                raise ValueError(f"Required timeseries key '{ts_key}' for '{key}' not found in data.")
            data_out[key] = ts_data

        for key in self.optional_data:
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
        g_sol_df = SolarGainsISO13790().generate(obj, data)  # Calls ISO 13709 conform norm directly

        return pd.concat([g_int_df, g_sol_df], axis=1, keys=["g_int", "g_sol"])


def calculate_timeseries_5r1c(
    meta: dict, series: dict, controls: pd.DataFrame, params: dict, splits: dict
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run the 5R1C simulation over the full index using the prepared bundle."""

    idx: pd.DatetimeIndex = meta["index"]
    dt_s: float = float(meta["dt_s"])
    weather: pd.DataFrame = meta[O.WEATHER]
    T_out_arr = weather[C.TEMP_AIR].to_numpy(dtype=float)

    n_len = len(idx)

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
    Cm = params[O.C_M]
    Htr_is = params[O.H_TR_IS]
    Htr_ms = params[O.H_TR_MS]
    Htr_w = params[O.H_TR_W]
    Htr_em = params[O.H_TR_EM]
    capacity_air = params[O.CAPACITANCE_AIR]

    # Splits
    sigma_surface = splits[O.SIGMA_SURFACE]
    sigma_conv = splits["sigma_conv"]
    f_conv = splits[O.FRAC_CONV_INTERNAL]
    f_rad_surf = splits[O.FRAC_RAD_SURFACE]
    f_rad_mass = splits[O.FRAC_RAD_MASS]
    A_m = splits[O.AREA_M]
    A_tot = splits[O.AREA_TOT]

    # Initial states (air and two masses)
    Ta = T_init
    Tm = T_init

    T_in_arr = np.empty(n_len, dtype=float)
    Qh_arr = np.empty(n_len, dtype=float)
    Qc_arr = np.empty(n_len, dtype=float)

    T_in_arr[0] = Ta
    Qh_arr[0] = 0.0
    Qc_arr[0] = 0.0

    # Precompute gain splits
    phi_ia_arr, phi_st_arr, phi_m_arr = _split_gains_5r1c(
        g_int_arr, g_sol_arr, f_conv, f_rad_surf, f_rad_mass, Htr_w=Htr_w, A_m=A_m, A_tot=A_tot
    )

    # Precompute optimization constants for constant ventilation case
    Y_tset_const = calc_y_tset(Cm, Htr_is, Htr_ms, Htr_w, Htr_em, sigma_surface, sigma_conv, dt_s)
    if np.allclose(Hve_vent_arr, Hve_vent_arr[0]) and np.allclose(Hve_inf_arr, Hve_inf_arr[0]):
        Y_phiset_const = calc_y_phiset(
            Cm, Htr_is, Htr_ms, Htr_w, Htr_em, Hve_inf_arr[0], Hve_vent_arr[0], capacity_air, dt_s
        )
    else:
        Y_phiset_const = None

    for t in range(1, n_len):
        T_out = T_out_arr[t]
        T_sup = T_sup_arr[t]

        Hve_vent = float(Hve_vent_arr[t])
        Hve_inf = float(Hve_inf_arr[t])

        phi_ia = float(phi_ia_arr[t])
        phi_st = float(phi_st_arr[t])
        phi_m = float(phi_m_arr[t])

        T_min = T_min_arr[t]
        T_max = T_max_arr[t]

        on_heat = on_heat_arr[t]
        on_cool = on_cool_arr[t]

        P_heat = P_heat_arr[t]
        P_cool = P_cool_arr[t]

        # 1) Free-float calculation (no HVAC)
        sol_free = solve_step_phiset_5r1c(
            Cm=Cm,
            Htr_is=Htr_is,
            Htr_ms=Htr_ms,
            Htr_w=Htr_w,
            Htr_em=Htr_em,
            Hve_inf=Hve_inf,
            Hve_vent=Hve_vent,
            capacity_air=capacity_air,
            T_e=T_out,
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
            Y=Y_phiset_const,
        )
        Ta_free = float(sol_free[1])
        Tm_free = float(sol_free[3])

        Q_applied = 0.0

        # 2) Decide whether we need heating or cooling based on free-float air temperature
        if Ta_free < T_min and on_heat:
            # Need heating to keep at lower bound (T_set = T_min)
            sol_tset = solve_step_tset_5r1c(
                Cm=Cm,
                Htr_is=Htr_is,
                Htr_ms=Htr_ms,
                Htr_w=Htr_w,
                Htr_em=Htr_em,
                Hve_inf=Hve_inf,
                Hve_vent=Hve_vent,
                capacity_air=capacity_air,
                T_e=T_out,
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
                Y=Y_tset_const,
            )
            Q_req = float(sol_tset[0])
            Q_applied, clamped = _clamp_hvac(Q_req, on_heat, on_cool, P_heat, P_cool)
            if clamped:
                # Recompute temperatures with capped power (Q_hc = P_heat)
                sol_phi = solve_step_phiset_5r1c(
                    Cm=Cm,
                    Htr_is=Htr_is,
                    Htr_ms=Htr_ms,
                    Htr_w=Htr_w,
                    Htr_em=Htr_em,
                    Hve_inf=Hve_inf,
                    Hve_vent=Hve_vent,
                    capacity_air=capacity_air,
                    T_e=T_out,
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
                    Y=Y_phiset_const,
                )
                Ta = float(sol_phi[1])  # < T_min
                Tm = float(sol_phi[3])
            else:
                Ta = float(sol_tset[1])  # == T_min
                Tm = float(sol_tset[3])
        elif Ta_free > T_max and on_cool:
            # Need cooling to keep at upper bound (T_set = T_max)
            sol_tset = solve_step_tset_5r1c(
                Cm=Cm,
                Htr_is=Htr_is,
                Htr_ms=Htr_ms,
                Htr_w=Htr_w,
                Htr_em=Htr_em,
                Hve_inf=Hve_inf,
                Hve_vent=Hve_vent,
                capacity_air=capacity_air,
                T_e=T_out,
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
                Y=Y_tset_const,
            )
            Q_req = float(sol_tset[0])
            Q_applied, clamped = _clamp_hvac(Q_req, on_heat, on_cool, P_heat, P_cool)
            if clamped:
                # Recompute temperatures with capped power (Q_hc = P_cool)
                sol_phi = solve_step_phiset_5r1c(
                    Cm=Cm,
                    Htr_is=Htr_is,
                    Htr_ms=Htr_ms,
                    Htr_w=Htr_w,
                    Htr_em=Htr_em,
                    Hve_inf=Hve_inf,
                    Hve_vent=Hve_vent,
                    capacity_air=capacity_air,
                    T_e=T_out,
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
                    Y=Y_phiset_const,
                )
                Ta = float(sol_phi[1])  # > T_max
                Tm = float(sol_phi[3])
            else:
                Ta = float(sol_tset[1])  # == T_max
                Tm = float(sol_tset[3])

        else:
            # Free-float stays within band or HVAC off
            Ta = Ta_free  # T_min < Ta_free < T_max
            Tm = Tm_free

        T_in_arr[t] = Ta
        Qh_arr[t] = max(Q_applied, 0.0)
        Qc_arr[t] = max(-Q_applied, 0.0)

    T_in = pd.Series(T_in_arr, index=idx, dtype=float)
    Q_heat = pd.Series(Qh_arr, index=idx, dtype=float)
    Q_cool = pd.Series(Qc_arr, index=idx, dtype=float)
    return T_in, Q_heat, Q_cool


def _split_gains_5r1c(
    g_int: np.ndarray,
    g_sol: np.ndarray,
    f_conv: float,
    f_rad_surf: float,
    f_rad_mass: float,
    Htr_w: Optional[float] = None,
    A_m: Optional[float] = None,
    A_tot: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split gains into (convective to air, radiant to surfaces, radiant to mass).

    If Htr_w, A_m and A_tot are provided, use ISO 13790-like split
    (Eureca-style): convective internal gains go to air, radiant + solar
    are distributed between mass and surfaces, with a window-related term.

    Otherwise, fall back to user-defined fractions.
    """
    g_int = np.asarray(g_int, dtype=float)
    g_sol = np.asarray(g_sol, dtype=float)

    # ISO-13790 split when enough information is provided
    if Htr_w is not None and A_m is not None and A_tot is not None and A_tot > 0.0:
        # Convective part of internal gains only
        g_int_conv = f_conv * g_int
        g_int_rad = (1.0 - f_conv) * g_int

        phi_ia = g_int_conv

        phi_rad_plus_sol = g_int_rad + g_sol  # radiant + solar

        phi_m = (A_m / A_tot) * phi_rad_plus_sol
        phi_st = (1.0 - A_m / A_tot - Htr_w / (9.1 * A_tot)) * phi_rad_plus_sol

        return phi_ia, phi_st, phi_m

    # Default: user-defined fractional split
    g_tot = g_int + g_sol
    phi_ia = f_conv * g_tot
    phi_rad = (1.0 - f_conv) * g_tot
    # `f_rad_surf + f_rad_mass` already normalized in `_prepare_inputs`
    phi_st = f_rad_surf * phi_rad
    phi_m = f_rad_mass * phi_rad
    return phi_ia, phi_st, phi_m


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


def solve_step_tset_5r1c(
    Cm: float,
    Htr_is: float,
    Htr_ms: float,
    Htr_w: float,
    Htr_em: float,
    Hve_inf: float,
    Hve_vent: float,
    T_e: float,
    T_sup: float,
    T_set: float,
    Tm_prev: float,
    Ta_prev: float,
    phi_ia: float,
    phi_st: float,
    phi_m: float,
    sigma_surface: float,
    sigma_conv: float,
    capacity_air: float,
    dt_s: float,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One-step 5R1C solve in setpoint mode (solve for required HVAC power).

    Returns array: [Q_hc_req, T_air=T_set, T_s, T_m].
    """
    # Unknowns: [Q_hc, T_s, T_m]
    q = np.zeros(3, dtype=np.float64)

    if Y is None:
        Y = calc_y_tset(Cm, Htr_is, Htr_ms, Htr_w, Htr_em, sigma_surface, sigma_conv, dt_s)

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


def calc_y_tset(
    Cm: float,
    Htr_is: float,
    Htr_ms: float,
    Htr_w: float,
    Htr_em: float,
    sigma_surface: float,
    sigma_conv: float,
    dt_s: float,
) -> np.ndarray:
    """Precompute the Y matrix for setpoint mode 5R1C solve."""
    Y = np.zeros((3, 3), dtype=np.float64)

    Y[0, 0] = sigma_conv
    Y[0, 1] = Htr_is

    Y[1, 0] = sigma_surface
    Y[1, 1] = -(Htr_is + Htr_w + Htr_ms)
    Y[1, 2] = Htr_ms

    Y[2, 1] = Htr_ms
    Y[2, 2] = -(Cm / dt_s + Htr_em + Htr_ms)

    return Y


def solve_step_phiset_5r1c(
    Cm: float,
    Htr_is: float,
    Htr_ms: float,
    Htr_w: float,
    Htr_em: float,
    Hve_inf: float,
    Hve_vent: float,
    T_e: float,
    T_sup: float,
    Tm_prev: float,
    Ta_prev: float,
    phi_ia: float,
    phi_st: float,
    phi_m: float,
    sigma_surface: float,
    sigma_conv: float,
    Q_hc: float,
    capacity_air: float,
    dt_s: float,
    Y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One-step 5R1C solve in applied-load mode (compute resultant temperatures).

    Returns array: [Q_hc, T_air, T_s, T_m].
    """
    q = np.zeros(3, dtype=np.float64)

    # Unknowns: [T_air, T_s, T_m]
    if Y is None:
        Y = calc_y_phiset(Cm, Htr_is, Htr_ms, Htr_w, Htr_em, Hve_inf, Hve_vent, capacity_air, dt_s)

    # Ventilation split in source term
    q[0] = -Q_hc * sigma_conv - (Hve_inf * T_e + Hve_vent * T_sup) - phi_ia - capacity_air * Ta_prev / dt_s
    q[1] = -Q_hc * sigma_surface - phi_st - Htr_w * T_e
    q[2] = -Htr_em * T_e - phi_m - Cm * Tm_prev / dt_s

    y = np.linalg.solve(Y, q)
    return np.array([Q_hc, y[0], y[1], y[2]], dtype=float)


def calc_y_phiset(
    Cm: float,
    Htr_is: float,
    Htr_ms: float,
    Htr_w: float,
    Htr_em: float,
    Hve_inf: float,
    Hve_vent: float,
    capacity_air: float,
    dt_s: float,
) -> np.ndarray:
    """Precompute the Y matrix for applied-load mode 5R1C solve."""
    Hve_tot = Hve_inf + Hve_vent
    Y = np.zeros((3, 3), dtype=np.float64)

    Y[0, 0] = -(Htr_is + Hve_tot) - capacity_air / dt_s
    Y[0, 1] = Htr_is

    Y[1, 0] = Htr_is
    Y[1, 1] = -(Htr_is + Htr_w + Htr_ms)
    Y[1, 2] = Htr_ms

    Y[2, 1] = Htr_ms
    Y[2, 2] = -(Cm / dt_s + Htr_em + Htr_ms)

    return Y
