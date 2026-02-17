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


class PHT(Method):
    """
    Derive binary occupancy from electricity demand using the Page–Hinkley Test (PHT).

    The method applies a Page–Hinkley change detection on the log-transformed power readings.
    It maintains a cumulative deviation statistic and its running extrema to detect upward
    and downward drifts. Detected upward drifts set occupancy to 1; downward drifts set it
    to 0. Optionally, a nightly schedule can force unoccupied states during specific hours.

    Notes:
        - The electricity time series supplies the index (timestamps) of the resulting
          occupancy series.
        - Power readings are log10-transformed with a small epsilon to avoid log(0).
        - The smoothing parameter (O.LAMBDA), baseline offset (O.BASELINE_OFFSET), and the
          detection threshold (O.DETECTION_THRESHOLD) control sensitivity and responsiveness.
    """

    types = [Types.OCCUPANCY]
    name = "PHT"
    required_keys = []
    required_data = [Types.ELECTRICITY]
    optional_keys = [
        O.DETECTION_THRESHOLD,
        O.BASELINE_OFFSET,
        O.NIGHT_SCHEDULE,
        O.NIGHT_SCHEDULE_START,
        O.NIGHT_SCHEDULE_END,
        O.LAMBDA,
    ]
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
        baseline_offset: float = None,
        detection_threshold: int = None,
        night_schedule: bool = None,
        night_schedule_start: int = None,
        night_schedule_end: int = None,
    ):
        """
        Generate a binary occupancy schedule from electricity demand using PHT.

        This method prepares inputs, applies defaults for optional parameters via the base
        Method utilities, delegates computation to calculate_timeseries, and formats the
        output (summary and timeseries) according to the framework conventions.

        Args:
            obj (dict, optional):
                Object parameters. Relevant keys (under the current method type) include:
                - O.LAMBDA (float): Exponential smoothing parameter used for the running average.
                - O.BASELINE_OFFSET (float): Baseline offset subtracted from deviations to
                  tune sensitivity (positive values reduce false positives).
                - O.DETECTION_THRESHOLD (float | int): Threshold for detecting drifts in the
                  Page–Hinkley statistic; larger values make detection less sensitive.
                - O.NIGHT_SCHEDULE (bool): Whether to enforce a nightly schedule.
                - O.NIGHT_SCHEDULE_START (int): Start hour of nightly off period [0-23].
                - O.NIGHT_SCHEDULE_END (int): End hour of nightly off period [0-23].
            data (dict, optional):
                Not used directly. The electricity time series is expected to be available in
                results under Types.ELECTRICITY.
            results (dict, optional):
                Dictionary with previously computed time series. Must contain an entry for
                Types.ELECTRICITY with key Keys.TIMESERIES that provides a pandas DataFrame
                with a datetime column Columns.DATETIME and at least one power column.
            ts_type (str, optional):
                Target time series type. Defaults to Types.OCCUPANCY.
            lambda_occ (float, optional): Smoothing factor for the running average used by PHT.
            baseline_offset (float, optional): Adjusts the deviation baseline.
            detection_threshold (int, optional): Sets the change detection sensitivity.
            night_schedule (bool, optional): If True, apply nightly zeroing.
            night_schedule_start (int, optional): Hour marking the start of the nightly off period.
            night_schedule_end (int, optional): Hour marking the end of the nightly off period.

        Returns:
            dict: A dictionary with two keys:
                - "summary" (dict): Contains aggregated indicators, including
                  f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY_AVG}" with the average occupancy.
                - "timeseries" (pd.DataFrame): A DataFrame indexed by datetime with one column
                  f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY}" containing 0/1 occupancy states.

        Raises:
            KeyError: If the required electricity time series is missing from results.
            ValueError: If the electricity data lacks a datetime column or any power column.

        Examples:
            >>> method = PHT()
            >>> out = method.generate(results={Types.ELECTRICITY: {K.TIMESERIES: elec_df}})
            >>> out["timeseries"].head()
        """

        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            lambda_occ=lambda_occ,
            baseline_offset=baseline_offset,
            detection_threshold=detection_threshold,
            night_schedule=night_schedule,
            night_schedule_start=night_schedule_start,
            night_schedule_end=night_schedule_end,
        )

        # Get input data
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, results, ts_type)

        occ_schedule = calculate_timeseries(processed_obj, processed_data)

        return self._format_output(occ_schedule, processed_data)

    def _get_input_data(
        self, obj: dict, data: dict, results: dict, method_type: str = Types.OCCUPANCY
    ) -> tuple[dict, dict]:
        obj_out = {
            O.ID: Method.get_with_backup(obj, O.ID),
            O.LAMBDA: Method.get_with_method_backup(obj, O.LAMBDA, method_type, Const.DEFAULT_LAMBDA.value),
            O.BASELINE_OFFSET: Method.get_with_method_backup(
                obj, O.BASELINE_OFFSET, method_type, Const.DEFAULT_LAMBDA.value
            ),
            O.DETECTION_THRESHOLD: Method.get_with_method_backup(
                obj, O.DETECTION_THRESHOLD, method_type, Const.DEFAULT_DETECTION_THRESHOLD.value
            ),
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
    """Compute occupancy schedule using the Page–Hinkley Test (PHT) method.

    Implements the core PHT change-detection logic on the log-transformed power readings.
    A cumulative deviation statistic is maintained alongside its running minimum and maximum
    to detect significant upward or downward drifts. Upon detecting an upward drift the
    occupancy state switches to 1; upon detecting a downward drift it switches to 0. An
    optional nightly schedule can enforce unoccupied states during specified hours.

    Args:
        obj (dict):
            Object configuration with required keys:
            - O.LAMBDA (float): Smoothing factor in (0, 1] for the running average of the
              log power signal.
            - O.BASELINE_OFFSET (float): Offset subtracted from the deviation to control
              sensitivity (acts like a penalty against declaring a change).
            - O.DETECTION_THRESHOLD (int | float): Threshold applied to the Page–Hinkley
              statistic to decide when a change is detected.
            - O.NIGHT_SCHEDULE (bool): If True, enforce nightly off period.
            - O.NIGHT_SCHEDULE_START (int): Start hour [0-23] of the nightly off period.
            - O.NIGHT_SCHEDULE_END (int): End hour [0-23] of the nightly off period.
        data (dict):
            Dictionary containing the electricity demand DataFrame at key Types.ELECTRICITY.
            The DataFrame must be indexed by timezone-aware datetimes and include a power
            column; common names are f"{Types.ELECTRICITY}{SEP}{C.POWER}" or C.POWER. If not
            found, the first column is used as a fallback and a warning is logged.

    Returns:
        pd.DataFrame: DataFrame indexed by datetime with a single integer column
        O.OCCUPANCY containing 0/1 occupancy states.

    Notes:
        - Power readings are log10-transformed using a small epsilon to avoid log(0).
        - The running average is updated recursively using lambda (lamda in code) as the
          smoothing factor.
        - After a change is detected, the cumulative statistic and its extrema are reset to
          enable subsequent detections.
    """

    electricity_demand = data[Types.ELECTRICITY]

    candidates = [f"{Types.ELECTRICITY}{SEP}{C.POWER}", C.POWER]
    elec_col = next((c for c in candidates if c in electricity_demand.columns), None)
    if elec_col is None:
        elec_col = next((col for col in electricity_demand.columns if C.POWER in col), None)
    if elec_col is None:
        elec_col = electricity_demand.columns[0]
        logger.warning(f"No column containing '{C.POWER}' found. Using '{elec_col}' as power column.")
    electricity_demand[C.POWER] = electricity_demand[elec_col]

    lamda = obj[O.LAMBDA]
    nightly_schedule = obj[O.NIGHT_SCHEDULE]
    baseline_offset = obj[O.BASELINE_OFFSET]
    detect_threshold = obj[O.DETECTION_THRESHOLD]

    eps = 1e-6
    x = np.log10(np.maximum(electricity_demand[C.POWER].astype(float).values, eps))

    mt = 0.0
    avg = x[0]
    inc_min = 0.0
    dec_max = 0.0
    state = 0

    schedule = np.zeros_like(x, dtype=int)

    for i in range(len(x)):
        avg = lamda * x[i] + (1 - lamda) * avg
        deviation = x[i] - avg - baseline_offset
        mt += deviation

        inc_min = min(inc_min, mt)
        dec_max = max(dec_max, mt)

        inc_pht = mt - inc_min  # lowest value mt has reached so far, used to detect upward drift
        dec_pht = dec_max - mt  # highest value mt has reached so far, used to detect downward drift

        if inc_pht > detect_threshold:
            state = 1
            mt = 0.0
            inc_min = 0.0
            dec_max = 0.0
        elif dec_pht > detect_threshold:
            state = 0
            mt = 0.0
            inc_min = 0.0
            dec_max = 0.0

        schedule[i] = state

    df_occ_schedule = pd.DataFrame(
        {O.OCCUPANCY: schedule},
        index=electricity_demand.index,
    )

    if nightly_schedule:
        df_occ_schedule = apply_nightly_schedule(
            df_occ_schedule,
            obj[O.NIGHT_SCHEDULE_START],
            obj[O.NIGHT_SCHEDULE_END],
        )

    return df_occ_schedule
