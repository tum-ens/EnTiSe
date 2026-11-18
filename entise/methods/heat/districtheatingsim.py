import logging
from typing import Optional

import numpy as np
import pandas as pd

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import output_index, site_zone
from entise.core.base import Method
from entise.methods.heat.utils import cleanup_try_dir, create_try_file
from entise.methods.utils._calendar import hourly_wall_series, read_profile, scale_to_energy, time_table, wall_years

logger = logging.getLogger(__name__)

DEFAULT_DEMAND_KWH = 1.0
DEFAULT_PROFILE_TYPE = "HEF03"
DEFAULT_DHW_SHARE = 0.0


class DistrictHeatSim(Method):
    """Space heating demand using DistrictHeatingSim’s BDEW methodology.

    Purpose and scope:
    - Wraps the DistrictHeatingSim BDEW implementation to synthesize hourly
      heating energy profiles for a selected BDEW profile type (e.g., HEF03),
      scaled so that the energy over the run equals ``Objects.DEMAND_KWH``.
    - Optionally splits the total load into space heating and DHW shares via
      the dhw_share parameter.

    Notes:
    - Provide weather with timestamps and air temperature. The model runs per
      calendar year on the hourly local wall clock of the object's ``timezone``
      (UTC if unset). On the 23-hour day of the change to summer time the
      skipped hour is not read; on the 25-hour day the repeated hour is read
      twice. Each row is the mean power over the interval that starts at its
      label.
    - The first three characters of profile_type define the building group, the
      last two its subtype, e.g., "HEF03".

    Reference:
    - DistrictHeatingSim (BDEW heat requirement):
      https://github.com/rl-institut/DistrictHeatingSim
    - BDEW guideline (German Association of Energy and Water Industries).
    """

    name = "districtheatingsim"
    types = [Types.HEATING]

    required_keys = [O.WEATHER]
    optional_keys = [O.DEMAND_KWH, O.PROFILE_TYPE, O.DHW_SHARE, O.TIMEZONE]

    required_data = [O.WEATHER]
    optional_data = []

    output_summary = {f"{Types.HEATING}{SEP}{C.DEMAND}[kWh]": "total heating demand"}

    output_timeseries = {
        f"{Types.HEATING}{SEP}{C.LOAD}[W]": "total heating load per timestep",
        f"{Types.HEATING}{SEP}load_space[W]": "space heating load per timestep (derived via dhw_share)",
        f"{Types.HEATING}{SEP}load_dhw[W]": "DHW load per timestep (derived via dhw_share)",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        results: dict = None,
        ts_type: str = Types.HEATING,
        *,
        annual_demand_kwh: float = None,
        weather: pd.DataFrame = None,
        profile_type: str = None,
        dhw_share: Optional[float] = None,
    ):
        """Generate a heating load timeseries using DistrictHeatingSim's BDEW heat model.

        This method prepares inputs (including optional overrides), computes the
        heating demand profile using DistrictHeatingSim’s BDEW implementation
        based on outdoor air temperature and profile parameters, and returns a
        summary and timeseries.

        Args:
            obj: Object dictionary with inputs like demand over the run, profile type,
                 DHW share, and weather key.
            data: Data dictionary containing the weather dataframe under O.WEATHER.
            results: Unused placeholder for interface compatibility.
            ts_type: Timeseries type, defaults to Types.HEATING.
            annual_demand_kwh: Optional heat demand in kWh over the simulated run (the
                keyword keeps its historical name).
            weather: Optional weather dataframe override with C.DATETIME and C.TEMP_AIR.
            profile_type: Optional BDEW profile type (e.g., "HEF03", "GBH01").
                          First three characters define building type, last two subtype.
            dhw_share: Optional domestic hot water share (0–1), passed to
                       DistrictHeatingSim as real_ww_share.

        Returns:
            dict: {"summary": {...}, "timeseries": DataFrame with columns:
              - "heating:load[W]" (total)
              - "heating:load_space[W]" (derived)
              - "heating:load_dhw[W]" (derived)
            indexed like the input weather datetimes.
        """
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            annual_demand_kwh=annual_demand_kwh,
            weather=weather,
            profile_type=profile_type,
            dhw_share=dhw_share,
        )
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

        ts_total_w = calculate_timeseries(processed_obj, processed_data)

        logger.debug(f"[districtheatsim heat]: Generating {ts_type} data")

        return self._format_output(ts_total_w, processed_obj, processed_data)

    def _get_input_data(self, obj, data, method_type=Types.HEATING):
        profile_type = Method.get_with_method_backup(obj, O.PROFILE_TYPE, method_type, DEFAULT_PROFILE_TYPE)
        profile_type = str(profile_type).strip().upper()

        if len(profile_type) < 4:
            raise ValueError("[districtheatsim] profile_type must be like 'GBH03', 'HEF04', 'GKO01'")

        # Normalize subtype to 2 digits (e.g., GKO1 -> GKO01)
        profile_type = f"{profile_type[:3]}{profile_type[3:].zfill(2)}"

        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.DEMAND_KWH: Method.get_with_method_backup(obj, O.DEMAND_KWH, method_type, DEFAULT_DEMAND_KWH),
            O.PROFILE_TYPE: profile_type,
            O.DHW_SHARE: Method.get_with_method_backup(obj, O.DHW_SHARE, method_type, DEFAULT_DHW_SHARE),
        }

        obj_out[O.TIMEZONE] = obj.get(O.TIMEZONE)

        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather, step, hours = time_table(data, weather_key, site_zone(obj_out))
        weather = Method._strip_weather_height(weather)
        data_out = {O.WEATHER: weather, "step": step, "hours": hours, "run_data": data}

        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        return obj_out, data_out

    @staticmethod
    def _format_output(ts_total_w: pd.Series, obj: dict, data: dict) -> dict:
        """Format output as EnTiSe {summary, timeseries} and (optionally) split total into space+DHW."""
        out_space = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
        out_dhw = f"{Types.DHW}{SEP}{C.LOAD}[W]"

        # dhw_share in [0, 1]
        dhw_share = obj.get(O.DHW_SHARE, DEFAULT_DHW_SHARE)
        dhw_share = max(0.0, min(1.0, dhw_share))

        # Split (derived outputs only)
        ts_total_w = ts_total_w.astype(float)
        ts_dhw_w = ts_total_w * dhw_share
        ts_space_w = ts_total_w - ts_dhw_w

        df = pd.DataFrame(
            {
                out_space: ts_space_w.round().astype(int),
                out_dhw: ts_dhw_w.round().astype(int),
            },
            index=output_index(ts_total_w.index, obj, data["run_data"]),
        )

        hours = data["hours"]
        demand_wh = int(round(float(np.dot(ts_total_w.to_numpy(), hours))))
        load_max_w = int(round(float(ts_total_w.max())))
        demand_dhw_wh = int(round(float(np.dot(ts_dhw_w.to_numpy(), hours))))
        load_dhw_max_w = int(round(float(ts_dhw_w.max())))

        summary = {
            f"{Types.HEATING}{SEP}{O.DEMAND}[Wh]": demand_wh,
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": load_max_w,
            f"{Types.DHW}{SEP}{O.DEMAND}[Wh]": demand_dhw_wh,
            f"{Types.DHW}{SEP}{O.LOAD_MAX}[W]": load_dhw_max_w,
        }
        return {"summary": summary, "timeseries": df}


def calculate_timeseries(obj, data):
    """Compute the heating load timeseries for the requested horizon.

    Uses DistrictHeatingSim’s BDEW implementation to generate an hourly-native
    heating demand profile for every calendar year that the run's local wall
    clock touches, from the hourly temperature of each local wall-clock hour.
    Each row of the run, which covers ``[t, t + step)``, takes the mean of the
    profile over the same span of the legal wall clock (see
    :mod:`entise.methods.utils._calendar`), and the result is scaled so that
    the energy over the run equals the demand.

    Args:
        obj: Processed object mapping with keys:
            - O.DEMAND_KWH: Heat demand in kWh over the simulated run (float).
            - O.PROFILE_TYPE: BDEW profile type string (e.g., "HEF03", "GBH01").
            - O.DHW_SHARE: Optional DHW share (0–1) forwarded to
              DistrictHeatingSim as real_ww_share (float, optional).
        data: Processed data mapping with the run's normalized O.WEATHER table
            (with C.TEMP_AIR) and its step.

    Returns:
        pandas.Series: Integer Watts indexed by the run's UTC instants.
    """
    try:
        from districtheatingsim.heat_requirement.heat_requirement_BDEW import calculate as bdew_calculate
    except ModuleNotFoundError as exc:
        if exc.name != "districtheatingsim":
            raise
        raise ImportError(
            "The districtheatingsim method requires a manual installation of DistrictHeatingSim "
            "with Python 3.11. See the EnTiSe installation guide."
        ) from exc

    demand_kwh = float(obj[O.DEMAND_KWH])

    profile_type = str(obj.get(O.PROFILE_TYPE, DEFAULT_PROFILE_TYPE)).strip().upper()
    profile_type = f"{profile_type[:3]}{profile_type[3:].zfill(2)}"

    building_type = profile_type[:3]
    subtype = profile_type[3:].zfill(2)

    dhw_share = obj.get(O.DHW_SHARE, None)
    dhw_share = float(dhw_share) if dhw_share is not None else None

    zone = site_zone(obj)
    weather = data[O.WEATHER]
    step = data["step"]
    axis = weather.index

    # Hourly temperature of each local wall-clock hour of every year the run touches.
    years = wall_years(axis, zone, step)
    temp_hour = hourly_wall_series(weather[C.TEMP_AIR].astype(float), zone, step)

    hourly_kwh_chunks: list[pd.Series] = []
    for y in years:
        idx_y = pd.date_range(pd.Timestamp(y, 1, 1), pd.Timestamp(y + 1, 1, 1), freq="h", inclusive="left")
        t_y = temp_hour.reindex(idx_y).interpolate(limit_direction="both")

        tmp_dir, tmp_try_path = create_try_file(idx_hour_full=idx_y, temperature=t_y, prefix=f"entise_dhs_try_{y}_")
        try:
            hourly_intervals, total, _, _, _ = bdew_calculate(
                JWB_kWh=float(demand_kwh),
                profiletype=building_type,
                subtype=subtype,
                TRY_file_path=tmp_try_path,
                year=int(y),
                real_ww_share=dhw_share,
            )
        finally:
            cleanup_try_dir(tmp_dir)

        index = pd.DatetimeIndex(pd.to_datetime(hourly_intervals))
        if index.tz is not None:
            index = index.tz_localize(None)
        hourly_kwh_chunks.append(pd.Series(np.asarray(total, dtype=float), index=index, name="kWh").sort_index())

    if not hourly_kwh_chunks:
        raise ValueError("[districtheatsim] No yearly outputs produced")

    # Hourly energy [kWh] is the mean power [kW] of the hour that starts at the label.
    profile_kw = pd.concat(hourly_kwh_chunks).sort_index()
    power = read_profile(profile_kw, axis, zone, step, name="districtheatingsim").clip(min=0.0)
    power = scale_to_energy(power, data["hours"], demand_kwh * 1000.0)
    return pd.Series(np.round(power).astype(int), index=axis, name=C.LOAD)
