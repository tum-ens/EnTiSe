import logging
from typing import Optional

import numpy as np
import pandas as pd

from entise._optional import require_module
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Constants as Const
from entise.constants import Objects as O
from entise.constants.constants import CP_AIR, RHO_AIR
from entise.core._time import RUN_TIME_KEY, output_index, run_time
from entise.core.base import Method
from entise.core.utils import align_auxiliary_series, resolve_ts_or_scalar, resolve_ts_or_scalar_by_id
from entise.methods.auxiliary.internal.selector import InternalGains
from entise.methods.auxiliary.solar.strategies import SolarGainsInactive, SolarGainsISO13790, SolarGainsTimeSeries
from entise.methods.auxiliary.ventilation.strategies import VentilationInactive, VentilationTimeSeries
from entise.methods.hvac._latent_cooling import compute_latent_cooling
from entise.methods.hvac._typed import TYPED, TypedViews
from entise.methods.hvac._weather import check_window_irradiance, hvac_weather, step_seconds
from entise.methods.hvac._windows import windows_table
from entise.methods.hvac.defaults import (
    DEFAULT_ACTIVE_COOLING,
    DEFAULT_ACTIVE_HEATING,
    DEFAULT_ACTIVE_INTERNAL_GAINS,
    DEFAULT_ACTIVE_SOLAR_GAINS,
    DEFAULT_ACTIVE_VENTILATION,
    DEFAULT_DEADBAND,
    DEFAULT_FRAC_CONV_INTERNAL,
    DEFAULT_FRAC_RAD_MASS,
    DEFAULT_FRAC_RAD_SURFACE,
    DEFAULT_GAINS_INTERNAL_LATENT,
    DEFAULT_POWER_COOLING,
    DEFAULT_POWER_HEATING,
    DEFAULT_SIGMA_5R1C,
    DEFAULT_TARGET_HUMIDITY_REL,
    DEFAULT_TEMP_INIT,
    DEFAULT_TEMP_MAX,
    DEFAULT_TEMP_MIN,
    DEFAULT_VENTILATION,
    DEFAULT_VENTILATION_SPLIT,
)

logger = logging.getLogger(__name__)


class R5C1(TypedViews, Method):
    """5R1C HVAC model aligned with ISO 13790's simplified dynamic method.

    Purpose and scope
        Captures key heat transfer paths between indoor air, internal
        surfaces, thermal mass, and exterior using five resistances and
        one aggregated capacitance (building mass). Better represents
        radiant/convective splits and envelope interactions than 1R1C,
        while remaining efficient for large-scale simulations.

    Conceptual structure
        Capacitance ``C_m`` (building thermal mass) exchanges with
        internal surfaces via ``H_tr,ms`` and with indoor air via
        ``H_tr,is``; windows and opaque elements couple to exterior via
        ``H_tr,w`` and ``H_tr,em``. Optional sky correction via
        ``H_tr,op,sky``. Internal and solar gains are split into
        radiant/convective parts and routed to air, surfaces, and mass
        using σ parameters. Ventilation losses are handled via ``H_ve``
        (scalar or timeseries).

    Notes
        Implements the ISO 13790 simplified dynamic method assumptions
        (lumped mass and linear heat transfer). Parameter mapping follows
        standard notation. For even richer transient behavior and phase
        shifts, consider a 7R2C model (see VDI 6007).

    Typed heating and cooling
        Select the model for ``heating``, ``cooling`` or both, instead of
        ``hvac``, to get typed results from one solve. Each typed result has
        the indoor temperature and the columns and summary keys of its type.
        The control of a type the model is not selected for is off. A
        type-scoped control input, such as ``heating:power_heating[W]``,
        applies to its own control. ``hvac`` and a typed key cannot select
        the same model on one object.

    Reference
        ISO 13790: Energy performance of buildings — Calculation of
        energy use for space heating and cooling (simplified dynamic
        method).
    """

    types = [Types.HVAC, Types.HEATING, Types.COOLING]
    name = "5R1C"
    required_keys = [
        O.H_TR_IS,
        O.H_TR_MS,
        O.H_TR_W,
        O.H_TR_EM,
        O.C_M,
        O.WEATHER,
    ]
    optional_keys = {
        O.TIMEZONE: None,
        # Controls
        O.POWER_HEATING: DEFAULT_POWER_HEATING,
        O.POWER_COOLING: DEFAULT_POWER_COOLING,
        O.ACTIVE_HEATING: DEFAULT_ACTIVE_HEATING,
        O.ACTIVE_COOLING: DEFAULT_ACTIVE_COOLING,
        O.ACTIVE_GAINS_INTERNAL: DEFAULT_ACTIVE_INTERNAL_GAINS,
        O.ACTIVE_GAINS_SOLAR: DEFAULT_ACTIVE_SOLAR_GAINS,
        O.ACTIVE_VENTILATION: DEFAULT_ACTIVE_VENTILATION,
        O.TEMP_INIT: DEFAULT_TEMP_INIT,
        O.TEMP_MIN: DEFAULT_TEMP_MIN,
        O.TEMP_MAX: DEFAULT_TEMP_MAX,
        O.DEADBAND: DEFAULT_DEADBAND,
        O.TARGET_HUMIDITY_REL: DEFAULT_TARGET_HUMIDITY_REL,
        O.GAINS_INTERNAL_LATENT: DEFAULT_GAINS_INTERNAL_LATENT,
        O.TEMP_SUPPLY: None,
        O.AREA: Const.DEFAULT_AREA.value,
        O.HEIGHT: Const.DEFAULT_HEIGHT.value,
        O.AREA_M: None,
        O.AREA_TOT: None,
        O.LAT: None,
        O.LON: None,
        # Ventilation
        O.H_VE: None,
        O.VENTILATION_SPLIT: DEFAULT_VENTILATION_SPLIT,
        # Sky facing surface
        O.H_TR_OP_SKY: None,
        # 5R1C splits
        O.SIGMA_SURFACE: DEFAULT_SIGMA_5R1C,
        # Gains splits
        O.FRAC_CONV_INTERNAL: DEFAULT_FRAC_CONV_INTERNAL,
        O.FRAC_RAD_SURFACE: DEFAULT_FRAC_RAD_SURFACE,
        O.FRAC_RAD_MASS: DEFAULT_FRAC_RAD_MASS,
        # Timeseries
        O.WINDOWS: None,
        O.GAINS_INTERNAL: None,
        O.GAINS_INTERNAL_COL: None,
        O.GAINS_SOLAR: None,
        O.VENTILATION: DEFAULT_VENTILATION,
        O.VENTILATION_COL: None,
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

    def generate(
        self,
        obj: dict | None = None,
        data: dict | None = None,
        results: dict | None = None,
        ts_type: str = Types.HVAC,
        **inputs,
    ) -> dict:
        """Generate the indoor temperature and the heating and cooling loads of the 5R1C model.

        Solar gains: the 5R1C model looks up the object's windows by its ``id``, so a
        direct call passes ``obj={"id": ...}`` unless ``active_gains_solar`` is False.

        Args:
            obj (dict, optional): Object keys. Defaults to None.
            data (dict, optional): Input tables. Defaults to None.
            results (dict, optional): Results of the time series already generated for the object.
            ts_type (str, optional): Time series type to generate. Defaults to Types.HVAC.
            **inputs: Any input of the method, by its key or by its Python name (the key
                without its unit, such as ``H_tr_is`` for ``H_tr_is[W K-1]``). Keywords override the same keys
                in ``obj``. A table input, such as ``weather``, can be a DataFrame.

        Returns:
            dict: ``"summary"`` with the demand and load statistics and ``"timeseries"``
            with the indoor temperature and the loads.
        """
        obj, data = self._process_kwargs(obj, data, ts_type, **inputs)
        if ts_type in TYPED:
            return self.group_result(obj, data, results, ts_type)
        require_module("hvac-5r1c", "pvlib")
        # Read the tables once for this call; every helper below reuses this run.
        data = {**data, RUN_TIME_KEY: run_time(data)}
        obj, tables = self._get_input_data(obj, data, ts_type)
        data = self._prepare_inputs(obj, tables)

        temp_in, p_heat, p_cool = calculate_timeseries_5r1c(**data)
        meta = data["meta"]
        p_cool_sensible, p_cool_latent = self._apply_latent_cooling(obj, data, p_cool)
        return self._format_output(
            temp_in.round(3),
            p_heat.round().astype(int),
            p_cool_sensible.round().astype(int),
            p_cool_latent.round().astype(int),
            output_index(meta["index"], obj, tables),
            meta["dt_s"],
        )

    @staticmethod
    def _apply_latent_cooling(obj: dict, data: dict, p_cool: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Run the shared latent-cooling post-pass (issue #103) on top of the
        sensible-only 5R1C solver output. When `active_cooling` is False or
        the weather lacks humidity/pressure, latent is zero and sensible is
        returned unchanged."""
        meta = data["meta"]
        controls = data["controls"]
        weather = meta[O.WEATHER]
        index = meta["index"]

        active_cool_series = controls[O.ACTIVE_COOLING]
        # If any timestep has active cooling, run the post-pass; the helper
        # itself masks per-step where sensible is zero anyway.
        if not bool(np.asarray(active_cool_series).astype(bool).any()):
            return p_cool, pd.Series(np.zeros(len(index), dtype=float), index=index)

        # Total-capacity cap, per-step. Same series used to clip sensible in
        # the solver — reinterpreted here as the total (sensible + latent) cap.
        p_cool_max = controls[O.POWER_COOLING].to_numpy(dtype=float, copy=False)

        # Ventilation: 5R1C carries H_ve split into mechanical + infiltration
        # frames; sum both to reconstruct total conductance.
        h_ve_df = data["series"]["ventilation"]
        h_ve = np.asarray(h_ve_df.sum(axis=1), dtype=float)

        gains_lat = resolve_ts_or_scalar(obj, data, O.GAINS_INTERNAL_LATENT, index).to_numpy(dtype=float, copy=False)

        # Cooling setpoint is a series in 5R1C; the target humidity uses the
        # per-step T_max as the reference dry-bulb.
        t_max_arr = controls[O.TEMP_MAX].to_numpy(dtype=float, copy=False)

        p_sens_out, p_lat_out = compute_latent_cooling(
            weather=weather,
            p_cool_sensible=np.asarray(p_cool, dtype=float),
            p_cool_max=p_cool_max,
            h_ve=h_ve,
            gains_internal_latent=gains_lat,
            temp_max_c=t_max_arr,
            target_humidity_rel=float(obj[O.TARGET_HUMIDITY_REL]),
        )
        return (
            pd.Series(p_sens_out, index=index),
            pd.Series(p_lat_out, index=index),
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
        obj_out = self.get_relevant_objects(obj, method_type)
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

        # Add data for computation
        data_out = {}

        # Create meta data
        weather = data[O.WEATHER]
        index = weather.index
        meta = {
            O.WEATHER: weather,
            "index": index,
            "dt_s": step_seconds(weather),
        }
        data_out["meta"] = meta

        # Gains (generate if active and missing)
        data_out["series"] = {}
        gains = self._compute_gains(obj, data)
        data_out["series"]["gains"] = gains

        # Ventilation → split into mechanical & infiltration.
        # Honor the ACTIVE_VENTILATION flag: when explicitly False, use the
        # zero-series path instead of computing ventilation. Applied here
        # (rather than in the Ventilation selector) because R5C1 bypasses
        # the selector to hit the ISO-13709-specific TimeSeries strategy.
        if not bool(obj.get(O.ACTIVE_VENTILATION, True)):
            ven_df = VentilationInactive().generate(obj, data)
        elif obj.get(O.H_VE) is not None:
            # A ventilation coefficient on the object takes the place of the ventilation input.
            ven_df = resolve_ts_or_scalar_by_id(obj, data, O.H_VE, index)
        else:
            ven_df = VentilationTimeSeries().generate(obj, data)
        # Align onto the model index before the split: the auxiliary carries the
        # user's table index, and a Series x Series product would otherwise align
        # on the union of the two and silently produce an over-long, NaN-padded
        # frame (issue #106).
        ven_df = align_auxiliary_series(ven_df, index, O.VENTILATION, self.name)
        vent_split = resolve_ts_or_scalar(obj, data, O.VENTILATION_SPLIT, index)
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
            O.TEMP_MIN: resolve_ts_or_scalar(obj, data, O.TEMP_MIN, index),
            O.TEMP_MAX: resolve_ts_or_scalar(obj, data, O.TEMP_MAX, index),
            O.TEMP_SUPPLY: temp_sup,
            O.POWER_HEATING: resolve_ts_or_scalar(obj, data, O.POWER_HEATING, index),
            O.POWER_COOLING: resolve_ts_or_scalar(obj, data, O.POWER_COOLING, index),
            O.ACTIVE_HEATING: resolve_ts_or_scalar(obj, data, O.ACTIVE_HEATING, index),
            O.ACTIVE_COOLING: resolve_ts_or_scalar(obj, data, O.ACTIVE_COOLING, index),
        }
        controls = pd.DataFrame(controls)
        data_out["controls"] = controls

        # Parameters
        volume = float(obj[O.AREA]) * float(obj[O.HEIGHT])
        capacity_air = volume * RHO_AIR * CP_AIR
        params = {
            O.TEMP_INIT: float(obj[O.TEMP_INIT]),
            O.C_M: float(obj[O.C_M]),
            O.H_TR_IS: float(obj[O.H_TR_IS]),
            O.H_TR_MS: float(obj[O.H_TR_MS]),
            O.H_TR_W: float(obj[O.H_TR_W]),
            O.H_TR_EM: float(obj[O.H_TR_EM]),
            O.CAPACITANCE_AIR: float(capacity_air),
            # Symmetric thermostat dead band. deadband = 0 collapses the
            # hysteresis state machine to aim-for-setpoint (bit-exact
            # regression against the pre-hysteresis solver).
            O.DEADBAND: float(obj[O.DEADBAND]),
        }
        data_out["params"] = params

        # Splits
        sigma_surface = float(obj[O.SIGMA_SURFACE])
        f_conv = float(obj[O.FRAC_CONV_INTERNAL])
        f_rad_surf = float(obj[O.FRAC_RAD_SURFACE])
        f_rad_mass = float(obj[O.FRAC_RAD_MASS])
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
    def _format_output(temp_in, p_heat, p_cool_sensible, p_cool_latent, index, dt_s) -> dict:
        # `cooling:load[W]` carries the TOTAL (sensible + latent) cooling load
        # per issue #103. When latent is zero everywhere, total == sensible
        # and the pre-latent regression is bit-preserved.
        p_cool_total = p_cool_sensible + p_cool_latent
        # Each row is the mean power over its interval, so energy is the sum times the step.
        E_h_Wh = float(p_heat.to_numpy(dtype=float).sum() * dt_s / 3600.0)
        E_c_Wh = float(p_cool_total.to_numpy(dtype=float).sum() * dt_s / 3600.0)
        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": E_h_Wh,
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": float(max(p_heat.max(), 0.0)),
            f"{Types.COOLING}{SEP}{C.DEMAND}[Wh]": E_c_Wh,
            f"{Types.COOLING}{SEP}{O.LOAD_MAX}[W]": float(max(p_cool_total.max(), 0.0)),
        }

        df = pd.DataFrame(
            {
                C.TEMP_IN: temp_in,
                f"{Types.HEATING}{SEP}{C.LOAD}[W]": p_heat,
                f"{Types.COOLING}{SEP}{C.LOAD}[W]": p_cool_total,
                f"{Types.COOLING}{SEP}sensible_{C.LOAD}[W]": p_cool_sensible,
                f"{Types.COOLING}{SEP}latent_{C.LOAD}[W]": p_cool_latent,
            },
            index=index,
        )

        return {"summary": summary, "timeseries": df}

    def _prepare_data_tables(self, obj: dict, data: dict, method_type: str = Types.HVAC) -> dict:
        run, weather_key, weather = hvac_weather(obj, data, method_type)
        data_out = {RUN_TIME_KEY: run, O.WEATHER: weather}
        check_window_irradiance(run, weather_key, weather, obj, windows_table(obj, data, method_type), method_type)

        for key in self.optional_data:
            ts_key = self.get_with_method_backup(obj, key, method_type)
            if key == O.WINDOWS:
                ts_data = windows_table(obj, data, method_type)
            else:
                ts_data = self.get_with_backup(data, ts_key)
            if ts_data is not None:
                data_out[key] = ts_data
                # Auxiliary strategies (VentilationTimeSeries, InternalTimeSeries)
                # look their table up by the *user's* key — the value of obj[key] —
                # not by the canonical key, so keep it reachable under both names
                # (issue #106). R1C1 files it under the user's key only.
                if isinstance(ts_key, str) and ts_key != key:
                    data_out.setdefault(ts_key, ts_data)

        return {k: v for k, v in data_out.items() if v is not None}

    @staticmethod
    def _compute_gains(obj: dict, data: dict) -> pd.DataFrame:
        g_int_df = InternalGains().generate(obj, data)
        # Honor the ACTIVE_GAINS_SOLAR flag: when explicitly False, use the
        # zero-series path instead of running the ISO 13790 solar computation.
        # Applied here (rather than in the SolarGains selector) because R5C1
        # bypasses the selector to hit the ISO-13790-specific strategy.
        if not bool(obj.get(O.ACTIVE_GAINS_SOLAR, True)):
            g_sol_df = SolarGainsInactive().generate(obj, data)
        elif obj.get(O.GAINS_SOLAR) is not None:
            g_sol_df = SolarGainsTimeSeries().generate(obj, data)
        else:
            g_sol_df = SolarGainsISO13790().generate(obj, data)
        return pd.concat([g_int_df, g_sol_df], axis=1, keys=["g_int", "g_sol"])


def calculate_timeseries_5r1c(
    meta: dict, series: dict, controls: pd.DataFrame, params: dict, splits: dict
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run the 5R1C simulation over the full index using the prepared bundle."""

    idx: pd.DatetimeIndex = meta["index"]
    dt_s: float = float(meta["dt_s"])
    weather: pd.DataFrame = meta[O.WEATHER]
    T_out_arr = weather[C.TEMP_AIR].to_numpy(dtype=float).ravel()

    n_len = len(idx)

    # Gains
    g_int_arr = series["gains"]["g_int"].to_numpy(dtype=float).ravel()
    g_sol_arr = series["gains"]["g_sol"].to_numpy(dtype=float).ravel()

    # Ventilation
    Hve_vent_arr = series["ventilation"]["Hve_vent"].to_numpy(dtype=float).ravel()
    Hve_inf_arr = series["ventilation"]["Hve_inf"].to_numpy(dtype=float).ravel()

    # Controls
    T_min_arr = controls[O.TEMP_MIN].to_numpy(dtype=float).ravel()
    T_max_arr = controls[O.TEMP_MAX].to_numpy(dtype=float).ravel()
    T_sup_arr = controls[O.TEMP_SUPPLY].to_numpy(dtype=float).ravel()

    P_heat_arr = controls[O.POWER_HEATING].to_numpy(dtype=float).ravel()
    P_cool_arr = controls[O.POWER_COOLING].to_numpy(dtype=float).ravel()

    on_heat_arr = controls[O.ACTIVE_HEATING].to_numpy(dtype=bool).ravel()
    on_cool_arr = controls[O.ACTIVE_COOLING].to_numpy(dtype=bool).ravel()

    # Parameters
    T_init = params[O.TEMP_INIT]
    Cm = params[O.C_M]
    Htr_is = params[O.H_TR_IS]
    Htr_ms = params[O.H_TR_MS]
    Htr_w = params[O.H_TR_W]
    Htr_em = params[O.H_TR_EM]
    capacity_air = params[O.CAPACITANCE_AIR]
    deadband = float(params.get(O.DEADBAND, DEFAULT_DEADBAND))

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

    # Hysteresis state — see issue #102 and the R1C1 solver for the
    # matching state-machine definition.
    heating_on = False
    cooling_on = False

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

    # Row t is the interval [t, t + step): it starts at T_in_arr[t], is driven by the
    # forcing of row t and ends at T_in_arr[t + 1].
    for t in range(n_len):
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

        # 2) Hysteresis-aware trigger. `heating_on` fires when either
        #    (a) heating was off and Ta_free dropped below T_min, or
        #    (b) heating was on and Ta_free is still below the upper band edge.
        # Cooling mirrors. When deadband == 0 the two branches collapse to
        # "fire iff Ta_free < T_min" and the "on" branch never fires more
        # than the "off" branch would — bit-exact regression.
        T_min_hi = T_min + deadband
        T_max_lo = T_max - deadband
        # Wide-band mutex: clear the opposite latch whenever the primary
        # extreme trigger of the other side fires. Physically we cannot heat
        # while already above T_max, nor cool while below T_min. With
        # deadband = 0 these guards are no-ops.
        if Ta_free > T_max:
            heating_on = False
        if Ta_free < T_min:
            cooling_on = False
        fire_h = on_heat and (Ta_free < T_min or (heating_on and Ta_free < T_min_hi))
        fire_c = on_cool and (Ta_free > T_max or (cooling_on and Ta_free > T_max_lo))

        if fire_h:
            # Aim for the upper band edge (T_min when deadband == 0).
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
                T_set=T_min_hi,
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
                Ta = float(sol_phi[1])  # < T_min_hi
                Tm = float(sol_phi[3])
            else:
                Ta = float(sol_tset[1])  # == T_min_hi
                Tm = float(sol_tset[3])
            heating_on = True
            cooling_on = False
        elif fire_c:
            # Aim for the lower band edge (T_max when deadband == 0).
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
                T_set=T_max_lo,
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
                Ta = float(sol_phi[1])  # > T_max_lo
                Tm = float(sol_phi[3])
            else:
                Ta = float(sol_tset[1])  # == T_max_lo
                Tm = float(sol_tset[3])
            heating_on = False
            cooling_on = True

        else:
            # Free-float stays within band or HVAC off
            Ta = Ta_free  # T_min < Ta_free < T_max (or actuator off/in-band)
            Tm = Tm_free
            heating_on = False
            cooling_on = False

        if t + 1 < n_len:
            T_in_arr[t + 1] = Ta
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
