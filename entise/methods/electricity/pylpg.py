import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base import Method
from entise.methods.electricity.utils import extract_hh_electricity_columns, run_year_chunk

logger = logging.getLogger(__name__)

# Module-level caches (per process)
_DTS_CACHE: dict[Any, pd.DataFrame] = {}


class PyLPG(Method):
    """PyLPG electricity demand generation (EnTiSe standard interface)."""

    name = "pylpg"
    types = [Types.ELECTRICITY]

    required_keys = [O.HOUSEHOLDS, O.OCCUPANTS_PER_HOUSEHOLD, O.DATETIMES]
    optional_keys = [O.ENERGY_INTENSITY]

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
        households: Optional[int] = None,
        occupants_per_household: Optional[int] = None,
        datetimes: Optional[pd.DataFrame] = None,
        energy_intensity: Optional[str] = None,
    ) -> dict:
        """Generate an electricity load timeseries using the PyLPG backend.

        This is the public entry point for the PyLPG electricity method. It accepts
        either an object/data mapping or keyword overrides, prepares inputs,
        executes PyLPG year-by-year to produce minute-resolution household
        electricity energies, aggregates them energy-conservingly to the requested
        timestep, converts to average power [W], and returns a summary and
        timeseries dataframe.

        Notes:
        - The temporal scaffold is derived from data[O.DATETIMES][C.DATETIME].
        - Timestep must be a multiple of 60 seconds (>= 60 s). Other steps are
          rejected because PyLPG natively produces minute energies.
        - The method enforces a constant, DST-safe wall-clock grid internally.
        - Final output index is aligned back to the original O.DATETIMES labels.

        Args:
            obj: Optional object dictionary providing inputs. Recognized keys:
                - O.ID: Optional identifier used in log messages.
                - O.HOUSEHOLDS: Number of households to simulate (int, required).
                - O.OCCUPANTS_PER_HOUSEHOLD: Occupants per household (int, required).
                - O.ENERGY_INTENSITY: Optional PyLPG energy intensity profile name.
                - O.DATETIMES: Optional override key for selecting the datetimes
                  timeseries from the data mapping.
            data: Data dictionary containing required timeseries. Must include an
                entry for O.DATETIMES that is a DataFrame with a C.DATETIME column
                of wall-clock timestamps (tz-naive or parseable strings).
            results: Unused placeholder for interface compatibility.
            ts_type: Timeseries type; defaults to Types.ELECTRICITY (ignored here).
            households: Keyword override for O.HOUSEHOLDS.
            occupants_per_household: Keyword override for O.OCCUPANTS_PER_HOUSEHOLD.
            datetimes: Keyword override providing the O.DATETIMES DataFrame.
            energy_intensity: Keyword override for O.ENERGY_INTENSITY (e.g., a
                PyLPG intensity scenario name).

        Returns:
            dict: A dictionary with keys:
                - "summary": Mapping with total electricity demand [Wh] and
                  maximum load [W] over the horizon.
                - "timeseries": DataFrame with one column
                  "ELECTRICITY|load[W]" containing integer Watts indexed like
                  the input datetimes.
        """
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            households=households,
            occupants_per_household=occupants_per_household,
            datetimes=datetimes,
            energy_intensity=energy_intensity,
        )

        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)

        logger.debug("[pylpg]: Generated successfully.")
        return self._format_output(ts, processed_data)

    def _get_input_data(self, obj: dict, data: dict, method_type: str = Types.ELECTRICITY) -> tuple[dict, dict]:
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.HOUSEHOLDS: Method.get_with_method_backup(obj, O.HOUSEHOLDS, method_type),
            O.OCCUPANTS_PER_HOUSEHOLD: Method.get_with_method_backup(obj, O.OCCUPANTS_PER_HOUSEHOLD, method_type),
            O.ENERGY_INTENSITY: Method.get_with_method_backup(obj, O.ENERGY_INTENSITY, method_type, None),
        }
        dts_key = Method.get_with_method_backup(obj, O.DATETIMES, method_type, O.DATETIMES)
        dts = Method.get_with_backup(data, dts_key)

        data_out = {O.DATETIMES: dts}

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

        dt_s = float((ts.index[1] - ts.index[0]).total_seconds())
        demand_wh = int(round(float(ts[out_col].sum()) * dt_s / 3600.0))
        load_max_w = int(round(float(ts[out_col].max())))

        summary = {
            f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]": demand_wh,
            f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]": load_max_w,
        }

        # Align to the wall-clock grid derived from O.DATETIMES
        ts.index = data[O.DATETIMES][C.DATETIME]
        ts.index.name = C.DATETIME

        return {"summary": summary, "timeseries": ts}


def calculate_timeseries(processed_obj: Dict[str, Any], processed_data: Dict[str, Any]) -> pd.DataFrame:
    """Compute the electricity load timeseries at the requested resolution using PyLPG.

    The function executes the PyLPG backend for all calendar years covered
    by the requested horizon, extracts household-level electricity outputs
    at minute resolution, aggregates them to the user-defined timestep in an
    energy-conserving manner, and converts the resulting energy per step
    into average power [W].

    The temporal scaffold is governed exclusively by O.DATETIMES:
    - Wall-clock timestamps are parsed without timezone shifting.
    - A constant, DST-safe grid is enforced via pd.date_range.
    - Multi-year horizons are split into yearly chunks, executed
      independently in PyLPG, and concatenated.

    Energy alignment rules:
    - PyLPG internally produces minute-resolution energy values.
    - For dt_s == 60 seconds: direct reindexing is applied.
    - For dt_s > 60 seconds: minute energies are summed over each
      timestep (energy-conserving aggregation).
    - The aggregated energy per timestep (kWh) is converted to
      average power [W] using: P = E * 3600 / dt_s.

    Args:
        processed_obj: Validated object mapping containing:
            - O.HOUSEHOLDS
            - O.OCCUPANTS_PER_HOUSEHOLD
            - optional O.ENERGY_INTENSITY
        processed_data: Validated data mapping containing
            O.DATETIMES dataframe with a C.DATETIME column.

    Returns:
        pandas.DataFrame: Single-column DataFrame of integer Watts
        indexed with tz-naive, regular wall-clock timestamps matching
        the input horizon.
    """
    # Get keys
    obj_id: str = str(processed_obj.get(O.ID, "obj"))
    households: int = int(processed_obj[O.HOUSEHOLDS])
    occ: int = int(processed_obj[O.OCCUPANTS_PER_HOUSEHOLD])
    ei_name: Optional[str] = processed_obj.get(O.ENERGY_INTENSITY)

    # Get data
    df_dts = processed_data[O.DATETIMES]

    # Get wall clock time (local time)
    idx = pd.to_datetime(df_dts[C.DATETIME].astype(str).str.slice(0, 19), errors="raise")
    dt_s = float((idx.iat[1] - idx.iat[0]).total_seconds())

    idx_user = pd.date_range(start=idx.iat[0], periods=len(idx), freq=pd.Timedelta(seconds=dt_s))
    idx_user = pd.DatetimeIndex(idx_user, name=C.DATETIME)

    if dt_s < 60.0 or (dt_s % 60.0) != 0.0:
        raise ValueError(f"[pylpg] Unsupported timestep {dt_s}s (must be >=60s and multiple of 60s)")

    # update cached df index so _format_output can reuse data[O.DATETIMES].index cleanly
    df_dts.index = idx_user
    processed_data[O.DATETIMES] = df_dts

    # Get years for which to compute
    years = sorted(pd.Index(idx_user.year).unique().astype(int).tolist())

    minute_chunks: List[pd.DataFrame] = []
    for y in years:
        idx_y = idx_user[idx_user.year == y]
        if len(idx_y) < 2:
            continue

        start = pd.Timestamp(idx_y[0])
        end = pd.Timestamp(idx_y[-1]) + pd.Timedelta(seconds=dt_s)

        logger.info("[pylpg] Running chunk id=%s year=%s start=%s end=%s", obj_id, y, start, end)

        df_min = run_year_chunk(
            obj_id=obj_id,
            year=int(y),
            start=start,
            end=end,
            households=households,
            occupants_per_household=occ,
            energy_intensity_name=ei_name,
        )
        minute_chunks.append(df_min)

    if not minute_chunks:
        raise ValueError("[pylpg] No chunks produced from provided datetimes")

    df_min_all = pd.concat(minute_chunks).sort_index()

    # PyLPG minute outputs are typically kWh/min per HH; extract electricity HH columns
    df_e_min = extract_hh_electricity_columns(df_min_all, households)
    df_e_min.index = pd.DatetimeIndex(df_e_min.index, name=C.DATETIME)

    # Align minute energies to user timestep (energy-conserving)
    if dt_s == 60.0:
        df_e_user = df_e_min.reindex(idx_user)
    else:
        minutes = int(dt_s // 60)
        df_e_user = df_e_min.resample(f"{minutes}min").sum().reindex(idx_user)

    df_e_user = df_e_user.fillna(0.0)

    # Convert energy per timestep (kWh) -> average power [W]
    e_kwh_step = df_e_user.sum(axis=1).astype(float)
    load_w = (e_kwh_step * 1000.0 * 3600.0 / dt_s).round()

    out = pd.DataFrame({"pylpg": load_w.astype(int)}, index=idx_user.copy())
    out.index = pd.DatetimeIndex(out.index, name=C.DATETIME)
    return out
