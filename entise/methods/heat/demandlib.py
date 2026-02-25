import logging
from typing import Optional

import pandas as pd
from demandlib import bdew

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base import Method
from entise.methods.utils.holidays import get_holidays

logger = logging.getLogger(__name__)

# Module-level caches (per process)
_WEATHER_CACHE: dict[tuple, pd.DataFrame] = {}

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
      are scaled to an annual demand target and aligned to the requested timestep.

    Notes:
    - Provide weather with a datetime column and air temperature. The method derives
      wall‑clock timestamps and applies an energy‑conserving resampling when your
      resolution differs from the native demandlib resolution.
    - Building type/class and wind class adjust the sensitivity and seasonal shape as
      per BDEW assumptions.

    Reference:
    - demandlib (BDEW heat): https://demandlib.readthedocs.io/
    - BDEW guideline (German Association of Energy and Water Industries).
    """

    name = "demandlib_heat"
    types = [Types.HEATING]

    required_keys = [O.WEATHER]
    optional_keys = [O.DEMAND_KWH, O.BUILDING_TYPE, O.BUILDING_CLASS, O.WIND_CLASS, O.HOLIDAYS_LOCATION]

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

        return self._format_output(ts, processed_data)

    def _get_input_data(self, obj, data, method_type=Types.HEATING):
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.DEMAND_KWH: Method.get_with_method_backup(obj, O.DEMAND_KWH, method_type, DEFAULT_DEMAND),
            O.BUILDING_TYPE: Method.get_with_method_backup(obj, O.BUILDING_TYPE, method_type, DEFAULT_BUILDING_TYPE),
            O.BUILDING_CLASS: Method.get_with_method_backup(obj, O.BUILDING_CLASS, method_type, DEFAULT_BUILDING_CLASS),
            O.WIND_CLASS: Method.get_with_method_backup(obj, O.WIND_CLASS, method_type, DEFAULT_WIND_CLASS),
            O.HOLIDAYS_LOCATION: Method.get_with_method_backup(obj, O.HOLIDAYS_LOCATION, method_type, None),
        }

        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather = Method.get_with_backup(data, weather_key)

        data_out = {
            O.WEATHER: weather,
        }

        if float(obj_out[O.DEMAND_KWH]) <= 0:
            raise ValueError("[demandlib_heat] demand must be > 0 ")

        # Clean up
        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        data_out = {k: v for k, v in data_out.items() if v is not None}

        # Safe datetime handling
        weather_cache_key = weather_key
        weather_cached = _WEATHER_CACHE.get(weather_cache_key)
        if weather_cached is None:
            if O.WEATHER in data_out:
                weather = data_out[O.WEATHER].copy()
                weather = self._strip_weather_height(weather)
                weather.index = pd.to_datetime(weather[C.DATETIME], utc=True)
                data_out[O.WEATHER] = weather
        else:
            data_out[O.WEATHER] = weather_cached

        return obj_out, data_out

    @staticmethod
    def _format_output(ts: pd.Series, data: dict) -> dict:
        out_col = f"{Types.HEATING}{SEP}{C.LOAD}[W]"
        df = ts.copy().round().astype(int).to_frame()
        df.columns = [out_col]
        df.index = pd.Index(data[O.WEATHER][C.DATETIME].values)
        df.index.name = C.DATETIME

        idx = ts.index.to_series()
        dt_s = (idx.iat[1] - idx.iat[0]).total_seconds()
        demand_wh = int(round(float(ts.sum()) * dt_s / 3600.0))
        load_max_w = int(round(float(ts.max())))

        summary = {
            f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]": demand_wh,
            f"{Types.HEATING}{SEP}{O.LOAD_MAX}[W]": load_max_w,
        }
        return {"summary": summary, "timeseries": df}


def calculate_timeseries(obj, data):
    """Compute the heating load timeseries for the requested horizon.

    Uses demandlib.bdew.HeatBuilding with given building parameters and
    temperature-driven SHLP to create an hourly-native profile, scales it to the
    requested annual demand, and aligns it to the target timestep. Subhourly
    inputs are aggregated to hourly for temperature, then interpolated back to
    the original dt_s using mean() and scaling to preserve power semantics.

    Args:
        obj: Processed object mapping with O.DEMAND_KWH, O.BUILDING_TYPE,
            O.BUILDING_CLASS, O.WIND_CLASS, and optional O.HOLIDAYS_LOCATION.
        data: Processed data mapping with O.WEATHER dataframe containing
            C.DATETIME (tz-aware/naive accepted; converted to tz-naive wall clock)
            and C.TEMP_AIR.

    Returns:
        pandas.Series of integer Watts indexed by tz-naive regular timestamps.
    """
    # Get keys
    demand_kwh = float(obj[O.DEMAND_KWH])
    building_type = str(obj[O.BUILDING_TYPE])
    building_class = int(obj[O.BUILDING_CLASS])
    wind_class = int(obj[O.WIND_CLASS])
    holidays_location = obj.get(O.HOLIDAYS_LOCATION, None)

    # Get data
    df_weather = data[O.WEATHER]

    # Get time delta
    idx_utc = df_weather.index
    dt_s = (idx_utc[1] - idx_utc[0]).total_seconds()
    steps_per_hour = 3600.0 / dt_s

    # Get local time
    df_weather[f"{C.DATETIME}_naive"] = pd.to_datetime(df_weather[f"{C.DATETIME}"].astype(str).str.slice(0, 19))

    # Make sure that local time is consistent (no DST changes)
    df_weather[f"{C.DATETIME}_naive"] = pd.date_range(
        start=df_weather[f"{C.DATETIME}_naive"].iloc[0], periods=len(df_weather), freq=pd.Timedelta(seconds=dt_s)
    )

    # Create air temperature time series with local time as index
    air_temp = df_weather.copy().set_index(f"{C.DATETIME}_naive")[C.TEMP_AIR].astype(float).rename("temperature")

    # Interpolate if dt_s is not 3600
    if dt_s != 3600.0:
        air_temp = air_temp.resample("1h").mean().interpolate()

    # Get holidays
    holidays = get_holidays(holidays_location=holidays_location, years=air_temp.index.year.unique())

    # Calculate heat demand
    heat = bdew.HeatBuilding(
        air_temp.index,
        shlp_type=building_type,
        building_class=building_class,
        wind_class=wind_class,
        holidays=holidays,
        temperature=air_temp,
        annual_heat_demand=demand_kwh,
    ).get_bdew_profile()

    # Ensure it was scaled properly and set to W
    heat = (heat / heat.sum()) * demand_kwh * 1000.0

    # Interpolate back if dt_s is not 3600
    if dt_s != 3600.0:
        heat = heat.resample(f"{dt_s}s").mean().interpolate() * steps_per_hour  # Multiplication since it is power

    return heat.round().astype(int)
