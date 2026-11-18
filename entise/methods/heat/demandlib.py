import logging
from typing import Optional

import numpy as np
import pandas as pd

from entise._optional import require_module, requires_extra
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import output_index, site_zone
from entise.core.base import Method
from entise.methods.utils._calendar import hourly_wall_series, read_profile, scale_to_energy, time_table, wall_years
from entise.methods.utils._demandlib import contain_warning_filters
from entise.methods.utils.holidays import get_holidays

logger = logging.getLogger(__name__)

# Default values for optional keys
DEFAULT_DEMAND = 1.0
DEFAULT_BUILDING_TYPE = "EFH"
DEFAULT_BUILDING_CLASS = 1
DEFAULT_WIND_CLASS = 0


class Demandlib(Method):
    """Space heating demand using demandlib’s temperature-driven BDEW model.

    Purpose and scope:
    - Wraps demandlib’s BDEW methodology to synthesize hourly heating energy profiles
      driven by outdoor air temperature and building archetype parameters. Profiles
      are scaled so that the energy over the run equals ``Objects.DEMAND_KWH``.

    Notes:
    - Provide weather with timestamps and air temperature. The engine runs on the
      hourly local wall clock of the object's ``timezone`` (UTC if unset). On the
      23-hour day of the change to summer time the skipped hour is not read; on
      the 25-hour day the repeated hour is read twice. Each row is the mean power
      over the interval that starts at its label.
    - Building type/class and wind class adjust the sensitivity and seasonal shape as
      per BDEW assumptions.

    Reference:
    - demandlib (BDEW heat): https://demandlib.readthedocs.io/
    - BDEW guideline (German Association of Energy and Water Industries).
    """

    name = "demandlib_heat"
    types = [Types.HEATING]

    required_keys = [O.WEATHER]
    optional_keys = [O.DEMAND_KWH, O.BUILDING_TYPE, O.BUILDING_CLASS, O.WIND_CLASS, O.HOLIDAYS_LOCATION, O.TIMEZONE]

    required_data = [O.WEATHER]
    optional_data = []

    output_summary = {f"{Types.HEATING}{SEP}{C.DEMAND}[kWh]": "total heating demand"}
    output_timeseries = {f"{Types.HEATING}{SEP}{C.LOAD}[W]": "heating load"}

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        results: dict = None,
        ts_type: str = Types.HEATING,
        *,
        annual_demand_kwh: float = None,
        weather: pd.DataFrame = None,
        building_type: str = None,
        building_class: int = None,
        wind_class: int = None,
        holidays_location: Optional[str] = None,
    ):
        """Generate a heating load timeseries using demandlib's BDEW heat model.

        This method prepares inputs (including optional overrides), computes the
        heating demand profile based on outdoor air temperature and BDEW
        parameters, and returns a summary and timeseries.

        Args:
            obj: Object dictionary with inputs like demand, building params, weather key.
            data: Data dictionary containing the weather dataframe under O.WEATHER.
            results: Unused placeholder for interface compatibility.
            ts_type: Timeseries type, defaults to Types.HEATING.
            annual_demand_kwh: Optional annual heat demand in kWh; defaults to 1.0.
            weather: Optional weather dataframe override with C.DATETIME and C.TEMP_AIR.
            building_type: Optional BDEW building type (e.g., "EFH").
            building_class: Optional BDEW building class (int).
            wind_class: Optional BDEW wind class (int).
            holidays_location: Optional country/region code for holidays (e.g., "DE").

        Returns:
            dict: {"summary": {...}, "timeseries": DataFrame with column
            "HEATING|load[W]" indexed like the input weather datetimes.
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            annual_demand_kwh=annual_demand_kwh,
            weather=weather,
            building_type=building_type,
            building_class=building_class,
            wind_class=wind_class,
            holidays_location=holidays_location,
        )
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)

        logger.debug(f"[demandlib heat]: Generating {ts_type} data")

        return self._format_output(ts, processed_obj, processed_data)

    def _get_input_data(self, obj, data, method_type=Types.HEATING):
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.DEMAND_KWH: Method.get_with_method_backup(obj, O.DEMAND_KWH, method_type, DEFAULT_DEMAND),
            O.BUILDING_TYPE: Method.get_with_method_backup(obj, O.BUILDING_TYPE, method_type, DEFAULT_BUILDING_TYPE),
            O.BUILDING_CLASS: Method.get_with_method_backup(obj, O.BUILDING_CLASS, method_type, DEFAULT_BUILDING_CLASS),
            O.WIND_CLASS: Method.get_with_method_backup(obj, O.WIND_CLASS, method_type, DEFAULT_WIND_CLASS),
            O.HOLIDAYS_LOCATION: Method.get_with_method_backup(obj, O.HOLIDAYS_LOCATION, method_type, None),
            O.TIMEZONE: obj.get(O.TIMEZONE),
        }

        if float(obj_out[O.DEMAND_KWH]) <= 0:
            raise ValueError("[demandlib_heat] demand must be > 0 ")

        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather, step = time_table(data, weather_key)
        weather = Method._strip_weather_height(weather)
        data_out = {O.WEATHER: weather, "step": step, "run_data": data}

        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        return obj_out, data_out

    @staticmethod
    def _format_output(ts: pd.Series, obj: dict, data: dict) -> dict:
        out_col = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
        df = ts.round().astype(int).to_frame(out_col)
        df.index = output_index(ts.index, obj, data["run_data"])

        hours = data["step"] / pd.Timedelta(1, unit="h")
        demand_wh = int(round(float(df[out_col].sum()) * hours))
        load_max_w = int(round(float(df[out_col].max())))

        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": demand_wh,
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": load_max_w,
        }
        return {"summary": summary, "timeseries": df}


def calculate_timeseries(obj, data):
    """Compute the heating load timeseries for the requested horizon.

    demandlib's ``HeatBuilding`` runs on an hourly grid of the site's local
    wall clock over whole local days, fed with the hourly mean air temperature
    of each wall-clock hour (the repeated hour is averaged and the skipped hour
    interpolated).
    Each row of the run, which covers ``[t, t + step)``, takes the mean of the
    hourly profile over the same span of the legal wall clock (see
    :mod:`entise.methods.utils._calendar`). The result is scaled so that the
    energy over the run equals the demand.

    Args:
        obj: Processed object mapping with O.DEMAND_KWH, O.BUILDING_TYPE,
            O.BUILDING_CLASS, O.WIND_CLASS, and optional O.HOLIDAYS_LOCATION
            and O.TIMEZONE.
        data: Processed data mapping with the run's normalized O.WEATHER table
            (with C.TEMP_AIR) and its step.

    Returns:
        pandas.Series of integer mean power [W] indexed by the run's UTC instants.
    """
    demand_kwh = float(obj[O.DEMAND_KWH])
    building_type = str(obj[O.BUILDING_TYPE])
    building_class = int(obj[O.BUILDING_CLASS])
    wind_class = int(obj[O.WIND_CLASS])
    holidays_location = obj.get(O.HOLIDAYS_LOCATION, None)
    zone = site_zone(obj)

    weather = data[O.WEATHER]
    step = data["step"]
    axis = weather.index

    # HeatBuilding works on daily mean temperatures, so it runs on whole local days.
    air_temp = hourly_wall_series(weather[C.TEMP_AIR].astype(float), zone, step, whole_days=True).rename("temperature")

    with requires_extra("heating-demandlib", "demandlib"):
        from demandlib import bdew
    require_module("heating-demandlib", "holidays")
    holidays = get_holidays(holidays_location=holidays_location, years=wall_years(axis, zone, step))

    with contain_warning_filters():
        profile = bdew.HeatBuilding(
            air_temp.index,
            shlp_type=building_type,
            building_class=building_class,
            wind_class=wind_class,
            holidays=holidays,
            temperature=air_temp,
            annual_heat_demand=demand_kwh,
        ).get_bdew_profile()

    power = read_profile(
        pd.Series(profile.to_numpy(dtype=float), index=air_temp.index), axis, zone, step, name="demandlib_heat"
    )
    power = scale_to_energy(power, step, demand_kwh * 1000.0)
    return pd.Series(np.round(power).astype(int), index=axis, name=C.LOAD)
