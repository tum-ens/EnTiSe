import logging
from typing import Any, Dict, Optional

import pandas as pd
from demandlib import bdew

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base import Method
from entise.methods.utils.holidays import get_holidays

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "h0_dyn"
DEFAULT_DEMAND = 1.0

# Module-level caches (per process)
_DTS_CACHE: dict[Any, pd.DataFrame] = {}


class Demandlib(Method):
    """Electricity demand generation using demandlib BDEW profiles."""

    name = "demandlib_electricity"
    types = [Types.ELECTRICITY]

    required_keys = [O.DATETIMES]
    optional_keys = [O.DEMAND_KWH, O.PROFILE, O.HOLIDAYS_LOCATION]

    required_timeseries = [O.DATETIMES]
    optional_timeseries = []

    output_summary = {
        f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]": "total electricity demand",
        f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]": "maximum electricity load",
    }
    output_timeseries = {
        f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]": "electricity load",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        results: dict = None,
        ts_type: str = Types.ELECTRICITY,
        *,
        profile: str = None,
        demand_kwh: float = None,
        weather: pd.DataFrame = None,
        holidays_location: Optional[str] = None,
    ) -> dict:
        """Generate an electricity demand timeseries using demandlib BDEW profiles.

        This is the public entry point for the electricity method. It accepts either
        an object/data mapping or keyword overrides, prepares inputs, computes the
        load profile at the requested resolution, and returns a summary and
        timeseries dataframe.

        Args:
            obj: Object dictionary providing inputs (e.g., demand, profile key).
            data: Data dictionary containing required timeseries (O.DATETIMES).
            results: Unused placeholder for interface compatibility.
            ts_type: Timeseries type, defaults to Types.ELECTRICITY.
            profile: Optional BDEW profile key (e.g., "h0", "g0"); defaults to
                method-level DEFAULT_PROFILE when not provided.
            demand_kwh: Optional annual demand in kWh; defaults to 1.0 if not provided.
            weather: Unused for this method; accepted for API symmetry.
            holidays_location: Optional holidays country/region code used by demandlib
                to adjust profiles (e.g., "DE").

        Returns:
            dict: A dictionary with keys:
                - "summary": Mapping with total demand [Wh] and max load [W].
                - "timeseries": DataFrame indexed like the input datetimes with one
                  column "ELECTRICITY|load[W]" of integer W values.
        """
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            profile=profile,
            demand_kwh=demand_kwh,
            weather=weather,
            holidays_location=holidays_location,
        )
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)

        logger.debug("[demandlib elec]: Generated succesfully.")

        return self._format_output(ts, processed_data)

    def _get_input_data(self, obj: dict, data: dict, method_type: str = Types.ELECTRICITY) -> tuple[dict, dict]:
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.PROFILE: Method.get_with_method_backup(obj, O.PROFILE, method_type, DEFAULT_PROFILE),
            O.DEMAND_KWH: Method.get_with_method_backup(obj, O.DEMAND_KWH, method_type, DEFAULT_DEMAND),
            O.HOLIDAYS_LOCATION: Method.get_with_method_backup(obj, O.HOLIDAYS_LOCATION, method_type, None),
        }

        dts_key = Method.get_with_method_backup(obj, O.DATETIMES, method_type, O.DATETIMES)
        dts = Method.get_with_backup(data, dts_key)

        data_out = {O.DATETIMES: dts}

        if float(obj_out[O.DEMAND_KWH]) <= 0:
            raise ValueError("[demandlib_electricity] demand must be > 0 ")

        # Clean up
        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        data_out = {k: v for k, v in data_out.items() if v is not None}

        # Safe datetime handling
        dts_cache_key = dts_key
        dts_cached = _DTS_CACHE.get(dts_cache_key)
        if dts_cached is None:
            if O.DATETIMES in data_out:
                dts = data_out[O.DATETIMES].copy()
                data_out[O.DATETIMES] = dts
                _DTS_CACHE[dts_key] = dts
        else:
            data_out[O.DATETIMES] = dts_cached

        return obj_out, data_out

    @staticmethod
    def _format_output(ts: pd.DataFrame, data: dict) -> dict:
        out_col = f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]"

        ts.columns = [out_col]

        dt_s = (ts.index[1] - ts.index[0]).total_seconds()
        demand_wh = int(round(float(ts[out_col].sum()) * dt_s / 3600.0))
        load_max_w = int(round(float(ts[out_col].max())))

        summary = {
            f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]": demand_wh,
            f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]": load_max_w,
        }

        ts.index = data[O.DATETIMES][C.DATETIME]

        return {"summary": summary, "timeseries": ts}


def calculate_timeseries(processed_obj: Dict[str, Any], processed_data: Dict[str, Any]) -> pd.DataFrame:
    """Compute the electricity load timeseries at the requested resolution.

    The function builds BDEW standard load profiles (SLPs) for all calendar years
    covered by the requested horizon, scales them to the specified annual demand
    (kWh), stitches multiple years if needed, and aligns the resulting 15-minute
    SLP to the target resolution using energy-conserving resampling rules:
    - For dt_s >= 900s (downsampling), use mean() to aggregate power correctly.
    - For dt_s < 900s (upsampling), use forward-fill to distribute step power.

    Args:
        processed_obj: Validated object mapping (contains O.PROFILE, O.DEMAND_KWH,
            and optional O.HOLIDAYS_LOCATION).
        processed_data: Validated data mapping with O.DATETIMES dataframe that has
            a C.DATETIME column.

    Returns:
        pandas.DataFrame: Single-column DataFrame of integer Watts indexed with
        tz-naive, regular wall-clock timestamps matching the input horizon.
    """
    # Get keys
    profile_type = processed_obj[O.PROFILE].lower()
    annual_demand_kwh = processed_obj[O.DEMAND_KWH]
    holidays_location = processed_obj.get(O.HOLIDAYS_LOCATION)

    # Get data
    df_dts = processed_data[O.DATETIMES]

    # Get wall clock time (local time)
    df_dts.index = pd.to_datetime(df_dts[C.DATETIME].astype(str).str.slice(0, 19))
    dt_s = (df_dts.index[1] - df_dts.index[0]).total_seconds()

    # Make sure that local time is consistent (no DST changes)
    df_dts.index = pd.date_range(start=df_dts.index[0], periods=len(df_dts), freq=pd.Timedelta(seconds=dt_s))

    # Get years for which to compute
    years = df_dts.index.year.unique().astype(int).sort_values()
    holidays = get_holidays(holidays_location=holidays_location, years=years)

    # Generate demand profile for every year that is required
    slps = []
    for y in years:
        slp = bdew.ElecSlp(y, holidays=holidays)
        # Get scaled profile profile
        demand = slp.get_scaled_power_profiles({profile_type: annual_demand_kwh})
        slps.append(demand)

    slps = pd.concat(slps)

    # Align the generated SLP years to the naive weather index
    # demandlib is 15min. If weather is 60min, mean() aggregates energy correctly.
    # If weather is 1min, reindex(method='ffill') distributes power correctly.
    if dt_s >= 900:  # 15 min or slower (Downsampling)
        output = slps.resample(f"{dt_s}s").mean().reindex(df_dts.index)
    else:  # Faster than 15 min (Upsampling)
        output = slps.reindex(df_dts.index, method="ffill")

    # Convert to W
    output *= 1000

    return output.round().astype(int)
