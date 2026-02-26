import logging
from typing import Optional

import numpy as np
import pandas as pd

# Try to import the optional dependency
try:
    from districtheatingsim.heat_requirement.heat_requirement_BDEW import calculate as bdew_calculate

    DISTRICTHEATINGSIM_AVAILABLE = True
except ImportError:
    DISTRICTHEATINGSIM_AVAILABLE = False
    bdew_calculate = None


from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base import Method
from entise.methods.heat.utils import cleanup_try_dir, create_try_file

logger = logging.getLogger(__name__)

# Module-level caches (per process)
_WEATHER_CACHE: dict[tuple, pd.DataFrame] = {}

DEFAULT_DEMAND_KWH = 1.0
DEFAULT_PROFILE_TYPE = "HEF03"
DEFAULT_DHW_SHARE = 0.0


class DistrictHeatSim(Method):
    """Space heating demand using DistrictHeatingSim’s BDEW methodology.

    Purpose and scope:
    - Wraps the DistrictHeatingSim BDEW implementation to synthesize hourly
      heating energy profiles for a selected BDEW profile type (e.g., HEF03),
      scaled to an annual demand target and aligned to the requested timestep.
    - Optionally splits the total load into space heating and DHW shares via
      the dhw_share parameter.

    Notes:
    - Provide weather with a datetime column and air temperature. The method
      derives wall‑clock timestamps and applies a power‑preserving alignment
      when your resolution differs from the hourly native resolution of the
      underlying model.
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
    optional_keys = [O.DEMAND_KWH, O.PROFILE_TYPE, O.DHW_SHARE]

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
            obj: Object dictionary with inputs like annual demand, profile type,
                 DHW share, and weather key.
            data: Data dictionary containing the weather dataframe under O.WEATHER.
            results: Unused placeholder for interface compatibility.
            ts_type: Timeseries type, defaults to Types.HEATING.
            annual_demand_kwh: Optional annual heat demand in kWh.
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

        return self._format_output(ts_total_w, processed_obj)

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

        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        weather = Method.get_with_backup(data, weather_key)

        data_out = {
            O.WEATHER: weather,
        }

        # Clean up (exactly like demandlib_heat)
        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        data_out = {k: v for k, v in data_out.items() if v is not None}

        # Safe datetime handling (exactly like demandlib_heat)
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
    def _format_output(ts_total_w: pd.Series, obj: dict) -> dict:
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
            index=pd.DatetimeIndex(ts_total_w.index),
        )
        df.index.name = C.DATETIME

        idx = ts_total_w.index.to_series()
        dt_s = (idx.iat[1] - idx.iat[0]).total_seconds()
        demand_wh = int(round(float(ts_total_w.sum()) * dt_s / 3600.0))
        load_max_w = int(round(float(ts_total_w.max())))
        demand_dhw_wh = int(round(float(ts_dhw_w.sum()) * dt_s / 3600.0))
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
    heating demand profile based on the specified BDEW profile type and
    temperature input. The annual demand is distributed over the computed
    hourly bins and converted to average hourly power. The resulting profile is
    then aligned to the target timestep. Subhourly inputs are aggregated to
    hourly temperature, and hourly average power is forward-filled back to the
    original dt_s to preserve power semantics.

    Args:
        obj: Processed object mapping with keys:
            - O.DEMAND_KWH: Annual heat demand in kWh (float).
            - O.PROFILE_TYPE: BDEW profile type string (e.g., "HEF03", "GBH01").
            - O.DHW_SHARE: Optional DHW share (0–1) forwarded to
              DistrictHeatingSim as real_ww_share (float, optional).
        data: Processed data mapping with O.WEATHER dataframe containing:
            - C.DATETIME: Datetime column (tz-aware/naive accepted; converted to
              tz-naive wall clock with a fixed step to avoid DST gaps/overlaps).
            - C.TEMP_AIR: Ambient air temperature in °C (numeric).

    Returns:
        pandas.Series: Integer Watts indexed by tz-naive regular timestamps.
    """
    demand_kwh = float(obj[O.DEMAND_KWH])

    profile_type = str(obj.get(O.PROFILE_TYPE, DEFAULT_PROFILE_TYPE)).strip().upper()
    profile_type = f"{profile_type[:3]}{profile_type[3:].zfill(2)}"

    building_type = profile_type[:3]
    subtype = profile_type[3:].zfill(2)

    dhw_share = obj.get(O.DHW_SHARE, None)
    dhw_share = float(dhw_share) if dhw_share is not None else None

    df_weather = data[O.WEATHER].copy()

    # Wall-clock index
    idx_wall = pd.to_datetime(df_weather[C.DATETIME].astype(str).str.slice(0, 19))
    dt_s = (idx_wall.iloc[1] - idx_wall.iloc[0]).total_seconds()
    idx_wall = pd.date_range(idx_wall.iloc[0], periods=len(idx_wall), freq=pd.Timedelta(seconds=dt_s))

    temp_hour = (
        df_weather.assign(_idx_wall=idx_wall)
        .set_index("_idx_wall")[C.TEMP_AIR]
        .astype(float)
        .resample("1h")
        .mean()
        .interpolate()
    )

    years = sorted(idx_wall.year.unique().astype(int).tolist())
    hourly_kwh_chunks: list[pd.Series] = []

    for y in years:
        start = pd.Timestamp(y, 1, 1)
        end = pd.Timestamp(y + 1, 1, 1)
        idx_y = pd.date_range(start, end, freq="h", inclusive="left", name=C.DATETIME)

        t_y = temp_hour.reindex(idx_y).interpolate(method="time")

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

        s_y = pd.Series(
            np.asarray(total, dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(hourly_intervals)).tz_localize(None),
            name="kWh",
        ).sort_index()
        hourly_kwh_chunks.append(s_y)

    if not hourly_kwh_chunks:
        raise ValueError("[districtheatsim] No yearly outputs produced")

    s_all_kwh = pd.concat(hourly_kwh_chunks).sort_index()

    # hourly kWh -> hourly average power [W]
    p_hour = (s_all_kwh * 1000.0).astype(float)

    if dt_s == 3600.0:
        p_aligned = p_hour.reindex(idx_wall)
    elif dt_s < 3600.0:
        p_aligned = p_hour.reindex(idx_wall, method="ffill")
    else:
        p_aligned = p_hour.resample(f"{int(dt_s)}s").mean().interpolate(method="time").reindex(idx_wall)

    p_aligned = p_aligned.clip(lower=0.0)
    p_aligned.index.name = C.DATETIME
    return p_aligned.round().astype(int)
