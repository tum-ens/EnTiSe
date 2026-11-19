import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import output_index, site_zone
from entise.core.base import Method
from entise.methods.electricity.utils import extract_hh_electricity_columns, run_year_chunk
from entise.methods.utils._calendar import read_profile, time_table, wall_window

logger = logging.getLogger(__name__)


class PyLPG(Method):
    """PyLPG electricity demand generation (EnTiSe standard interface).

    The Load Profile Generator simulates a plain local calendar in minutes. EnTiSe
    runs it over the local wall-clock span of the run in the object's ``timezone``
    (UTC if unset). Each row is the mean power over the interval that starts at
    its label. On the 23-hour day of the change to summer time the skipped hour is
    not read; on the 25-hour day the repeated hour is read twice.

    The engine draws its own random numbers, and EnTiSe does not seed it: the run
    seed of ``Generator.generate`` does not make PyLPG profiles reproducible.
    """

    name = "pylpg"
    types = [Types.ELECTRICITY]

    required_keys = [O.HOUSEHOLDS, O.OCCUPANTS_PER_HOUSEHOLD, O.DATETIMES]
    optional_keys = [O.ENERGY_INTENSITY, O.TIMEZONE]

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
        - The instants come from the O.DATETIMES table, and the local calendar
          from the object's ``timezone`` (UTC if unset).
        - Timestep must be a multiple of 60 seconds (>= 60 s). Other steps are
          rejected because PyLPG natively produces minute energies.

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
        return self._format_output(ts, processed_obj, processed_data)

    def _get_input_data(self, obj: dict, data: dict, method_type: str = Types.ELECTRICITY) -> tuple[dict, dict]:
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.HOUSEHOLDS: Method.get_with_method_backup(obj, O.HOUSEHOLDS, method_type),
            O.OCCUPANTS_PER_HOUSEHOLD: Method.get_with_method_backup(obj, O.OCCUPANTS_PER_HOUSEHOLD, method_type),
            O.ENERGY_INTENSITY: Method.get_with_method_backup(obj, O.ENERGY_INTENSITY, method_type, None),
            O.TIMEZONE: obj.get(O.TIMEZONE),
        }
        dts_key = Method.get_with_method_backup(obj, O.DATETIMES, method_type, O.DATETIMES)
        table, step, hours = time_table(data, dts_key, site_zone(obj_out))
        data_out = {O.DATETIMES: table, "step": step, "hours": hours, "run_data": data}

        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        return obj_out, data_out

    @staticmethod
    def _format_output(ts: pd.DataFrame, obj: dict, data: dict) -> dict:
        out_col = f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]"
        ts.columns = [out_col]

        demand_wh = int(round(float(np.dot(ts[out_col].to_numpy(dtype=float), data["hours"]))))
        load_max_w = int(round(float(ts[out_col].max())))

        summary = {
            f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]": demand_wh,
            f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]": load_max_w,
        }

        ts.index = output_index(ts.index, obj, data["run_data"])

        return {"summary": summary, "timeseries": ts}


def calculate_timeseries(processed_obj: Dict[str, Any], processed_data: Dict[str, Any]) -> pd.DataFrame:
    """Compute the electricity load timeseries at the requested resolution using PyLPG.

    The function executes the PyLPG backend for all calendar years covered
    by the requested horizon, extracts household-level electricity outputs
    at minute resolution, aggregates them to the user-defined timestep in an
    energy-conserving manner, and converts the resulting energy per step
    into average power [W].

    LPG simulates a plain local calendar. EnTiSe runs it over the local
    wall-clock span of the run, one calendar year at a time, and each row of
    the run, which covers ``[t, t + step)``, takes the mean power of the same
    span of the site's legal wall clock (see :mod:`entise.methods.utils._calendar`).
    On the 23-hour day the skipped hour is not read; on the 25-hour day the
    repeated hour is read twice.

    Args:
        processed_obj: Validated object mapping containing:
            - O.HOUSEHOLDS
            - O.OCCUPANTS_PER_HOUSEHOLD
            - optional O.ENERGY_INTENSITY
        processed_data: Mapping with the run's normalized O.DATETIMES table and its step.

    Returns:
        pandas.DataFrame: Single-column DataFrame of integer Watts indexed
        by the run's UTC instants.
    """
    obj_id: str = str(processed_obj.get(O.ID, "obj"))
    households: int = int(processed_obj[O.HOUSEHOLDS])
    occ: int = int(processed_obj[O.OCCUPANTS_PER_HOUSEHOLD])
    ei_name: Optional[str] = processed_obj.get(O.ENERGY_INTENSITY)
    zone = site_zone(processed_obj)

    axis = processed_data[O.DATETIMES].index
    step = processed_data["step"]
    dt_s = step / pd.Timedelta(1, unit="s")
    if dt_s < 60.0 or (dt_s % 60.0) != 0.0:
        raise ValueError(f"[pylpg] Unsupported timestep {dt_s}s (must be >=60s and multiple of 60s)")

    # LPG simulates a plain local calendar in minutes. Run it over the local wall-clock span of the run,
    # one calendar year at a time.
    start, end = wall_window(axis, zone, step)
    start, end = start.floor("min"), end.ceil("min")

    minute_chunks: List[pd.DataFrame] = []
    for year in range(start.year, (end - pd.Timedelta(1, unit="us")).year + 1):
        chunk_start = max(start, pd.Timestamp(year, 1, 1))
        chunk_end = min(end, pd.Timestamp(year + 1, 1, 1))
        logger.info("[pylpg] Running chunk id=%s year=%s start=%s end=%s", obj_id, year, chunk_start, chunk_end)
        minute_chunks.append(
            run_year_chunk(
                obj_id=obj_id,
                year=year,
                start=chunk_start,
                end=chunk_end,
                households=households,
                occupants_per_household=occ,
                energy_intensity_name=ei_name,
            )
        )

    df_min_all = pd.concat(minute_chunks).sort_index()

    # Minute energies [kWh] per household; their sum is the mean power of each minute.
    energy_kwh = extract_hh_electricity_columns(df_min_all, households).sum(axis=1)
    minutes = pd.date_range(start, end, freq="min", inclusive="left")
    energy_kwh = energy_kwh.groupby(level=0).sum()
    missing = minutes.difference(energy_kwh.index)
    if len(missing):
        raise ValueError(
            f"[pylpg] The engine returned no value for {len(missing)} minutes of local time, "
            f"the first at {missing[0]:%Y-%m-%d %H:%M}."
        )
    power_w = energy_kwh.reindex(minutes) * 60_000.0

    load_w = read_profile(power_w, axis, zone, step, name="pylpg")
    return pd.DataFrame({"pylpg": np.round(load_w).astype(int)}, index=axis)
