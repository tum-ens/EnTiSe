import logging

import numpy as np
import pandas as pd

from entise.constants import Columns as C
from entise.constants import Constants as Const
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.constants import Types
from entise.constants.general import SEP
from entise.core.base import Method

from .utils import apply_nightly_schedule

logger = logging.getLogger(__name__)


class GeoMA(Method):
    """
    Binary occupancy inference from electricity demand via a Geometric Moving Average (GeoMA).

    Purpose and scope:
    - Compares the log‑transformed instantaneous power against its exponentially weighted
      moving average (EWM). If the current reading exceeds the EWM (geometric mean on the
      original scale), occupancy=1 else 0. An optional nightly schedule can force unoccupied
      states during specified hours.

    Notes:
    - Electricity demand supplies the timestamp index; no separate clock needed.
    - Uses log10 with a small epsilon to avoid log(0) and stabilize low values.
    - Smoothing parameter ``Objects.LAMBDA`` tunes responsiveness of the EWM.

    Related methods:
    - See also PHT (Page–Hinkley Test) for change‑point based detection on the same input.
    """

    types = [Types.OCCUPANCY]
    name = "GeoMA"
    required_keys = []
    required_data = [Types.ELECTRICITY]
    optional_keys = [O.LAMBDA, O.NIGHT_SCHEDULE, O.NIGHT_SCHEDULE_START, O.NIGHT_SCHEDULE_END]
    output_summary = {
        f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}": "average occupancy",
    }
    output_timeseries = {
        f"{Types.OCCUPANCY}{SEP}{C.OCCUPANCY}": "binary occupancy schedule (0/1)",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        results: dict = None,
        ts_type: str = Types.OCCUPANCY,
        *,
        lambda_occ: float = None,
        night_schedule: bool = None,
        night_schedule_start: int = None,
        night_schedule_end: int = None,
    ):
        """
        Generate a binary occupancy schedule from electricity demand using GeoMA.

        This method is a thin orchestrator around calculate_timeseries. It prepares inputs,
        applies defaults for optional parameters (via the Method helpers), and formats the
        output as expected by the framework (summary and timeseries).

        Args:
            obj (dict, optional):
                Object parameters. Relevant keys (under the current method type) include:
                - O.LAMBDA (float): Exponential smoothing parameter in (0, 1].
                - O.NIGHT_SCHEDULE (bool): Whether to enforce a nightly schedule.
                - O.NIGHT_SCHEDULE_START (int): Start hour of nightly off period [0-23].
                - O.NIGHT_SCHEDULE_END (int): End hour of nightly off period [0-23].
            data (dict, optional):
                Not used directly for this method. The electricity time series is expected
                to be present in results under Types.ELECTRICITY.
            results (dict, optional):
                Dictionary with previously computed time series. Must contain an entry for
                Types.ELECTRICITY with key Keys.TIMESERIES that provides a pandas DataFrame
                with a datetime column Columns.DATETIME and at least one power column.
            ts_type (str, optional):
                Target time series type. Defaults to Types.OCCUPANCY.
            lambda_occ (float, optional): Exponential smoothing factor for the EWM used
                to compute the geometric moving average. Typical range (0.05–0.5).
            night_schedule (bool, optional): If True, apply nightly zeroing.
            night_schedule_start (int, optional): Hour of day marking the start of
                the nightly off period (e.g., 18:00).
            night_schedule_end (int, optional):  Hour of day marking the end of the
                nightly off period (e.g., 00:00).

        Returns:
            dict: A dictionary with two keys:
                - "summary" (dict): Contains aggregated indicators, including
                  f"{Types.OCCUPANCY}{SEP}{Objects.OCCUPANCY_AVG}" with the average occupancy.
                - "timeseries" (pd.DataFrame): A DataFrame indexed by datetime with one column
                  f"{Types.OCCUPANCY}{SEP}{Objects.OCCUPANCY}" containing 0/1 occupancy states.

        Raises:
            KeyError: If the required electricity time series is missing from results.
            ValueError: If the electricity data lacks a datetime column or any power column.

        Examples:
            >>> method = GeoMA()
            >>> out = method.generate(results={Types.ELECTRICITY: {K.TIMESERIES: elec_df}})
            >>> out["timeseries"].head()
        """

        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            lambda_occ=lambda_occ,
            night_schedule=night_schedule,
            night_schedule_start=night_schedule_start,
            night_schedule_end=night_schedule_end,
        )

        # Get input data
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, results, ts_type)

        # Compute temperature and energy demand
        occ_schedule = calculate_timeseries(processed_obj, processed_data)

        return self._format_output(occ_schedule, processed_data)

    def _get_input_data(
        self, obj: dict, data: dict, results: dict, method_type: str = Types.OCCUPANCY
    ) -> tuple[dict, dict]:
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.LAMBDA: Method.get_with_method_backup(obj, O.LAMBDA, method_type, Const.DEFAULT_LAMBDA.value),
            O.NIGHT_SCHEDULE: Method.get_with_method_backup(
                obj, O.NIGHT_SCHEDULE, method_type, Const.DEFAULT_NIGHT_SCHEDULE.value
            ),
            O.NIGHT_SCHEDULE_START: Method.get_with_method_backup(
                obj, O.NIGHT_SCHEDULE_START, method_type, Const.DEFAULT_NIGHT_SCHEDULE_START.value
            ),
            O.NIGHT_SCHEDULE_END: Method.get_with_method_backup(
                obj, O.NIGHT_SCHEDULE_END, method_type, Const.DEFAULT_NIGHT_SCHEDULE_END.value
            ),
        }

        # Store required results
        data_out = {k: results[k][K.TIMESERIES] for k in self.required_data if k in results}

        # Clean up
        obj_out = {k: v for k, v in obj_out.items() if v is not None}
        data_out = {k: v for k, v in data_out.items() if v is not None}

        # Safe datetime handling
        if Types.ELECTRICITY in data_out:
            electricity_demand = data_out[Types.ELECTRICITY].copy()
            electricity_demand[C.DATETIME] = pd.to_datetime(electricity_demand[C.DATETIME], utc=True)
            electricity_demand.set_index(C.DATETIME, inplace=True, drop=True)
            data_out[Types.ELECTRICITY] = electricity_demand

        return obj_out, data_out

    @staticmethod
    def _format_output(occ_schedule, data):
        summary = {
            f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}": round(occ_schedule[O.OCCUPANCY].mean(), 2),
        }

        df = pd.DataFrame(
            {f"{Types.OCCUPANCY}{SEP}{C.OCCUPANCY}": occ_schedule[O.OCCUPANCY]},
            index=data[Types.ELECTRICITY].index,
        )
        df.index.name = C.DATETIME

        return {"summary": summary, "timeseries": df}


def calculate_timeseries(obj: dict, data: dict) -> pd.DataFrame:
    """Compute occupancy schedule using the Geometric Moving Average (GeoMA) method.

    This function implements the core logic of the GeoMA approach used by GeoMA.generate.
    It log-transforms the power readings, computes an exponential moving average (EWM)
    with smoothing parameter Objects.LAMBDA, and marks time steps as occupied (1) if the
    current log power is greater than or equal to the EWM, otherwise unoccupied (0).
    Optionally, a nightly schedule can be applied to force unoccupied states during a
    specified off period.

    Args:
        obj (dict):
            Object configuration with required keys:
            - Objects.LAMBDA (float): Smoothing factor alpha for the EWM, in (0, 1].
            - Objects.NIGHT_SCHEDULE (bool): If True, enforce nightly off period.
            - Objects.NIGHT_SCHEDULE_START (int): Start hour [0-23] of nightly off period.
            - Objects.NIGHT_SCHEDULE_END (int): End hour [0-23] of nightly off period.
        data (dict):
            Dictionary containing the electricity demand DataFrame at key Types.ELECTRICITY.
            The DataFrame must be indexed by timezone-aware datetimes and include a power
            column; common names are f"{Types.ELECTRICITY}{SEP}{Columns.POWER}" or
            Columns.POWER. If not found, the first column is used as a fallback and a warning
            is logged.

    Returns:
        pd.DataFrame: DataFrame indexed by datetime with a single integer column
        O.OCCUPANCY containing 0/1 occupancy states.

    Notes:
        - Power readings are converted to float, clipped with a small epsilon, and log10-
          transformed for numerical stability and relative change sensitivity.
        - The EWM is computed with adjust=False to mimic recursive filtering.
    """

    electricity_demand = data[Types.ELECTRICITY]

    # Find correct column with power
    candidates = [f"{Types.ELECTRICITY}{SEP}{C.POWER}", C.POWER]
    elec_col = next((c for c in candidates if c in electricity_demand.columns), None)
    if elec_col is None:
        elec_col = next((col for col in electricity_demand.columns if C.POWER in col), None)
    if elec_col is None:
        elec_col = electricity_demand.columns[0]  # Fallback to the first column if no power column is found
        logger.warning(f"No column containing '{C.POWER}' found. Using '{elec_col}' as power column.")
    electricity_demand[C.POWER] = electricity_demand[elec_col]

    lamda = obj[O.LAMBDA]
    nightly_schedule = obj[O.NIGHT_SCHEDULE]

    eps = 1e-6
    log_readings = np.log10(np.maximum(electricity_demand[C.POWER].astype(float), eps)).round(3)

    # Calculate geometric moving average
    geoma_series = log_readings.ewm(alpha=lamda, adjust=False).mean().round(3)

    # Vectorized comparison
    schedule = (log_readings >= geoma_series).astype(int)

    df_occ_schedule = pd.DataFrame({O.OCCUPANCY: schedule}, index=electricity_demand.index)

    if nightly_schedule:
        df_occ_schedule = apply_nightly_schedule(
            df_occ_schedule, obj[O.NIGHT_SCHEDULE_START], obj[O.NIGHT_SCHEDULE_END]
        )

    return df_occ_schedule
