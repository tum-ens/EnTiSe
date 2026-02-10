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
    Geometric Moving Average method to derive occupancy from the electricity consumption data, it assigns occupancy
    whenever the average of the current measure is higher than the accumulated average, and vice versa.
    It requires the electricity time series to index the occupancy timeseries.
    """

    types = [Types.OCCUPANCY]
    name = "GeoMA"
    required_keys = []
    required_data = [Types.ELECTRICITY]
    optional_keys = [O.LAMBDA, O.NIGHT_SCHEDULE, O.NIGHT_SCHEDULE_START, O.NIGHT_SCHEDULE_END]

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        results: dict = None,
        ts_type: str = Types.OCCUPANCY,
        *,
        lambda_geoma: float = None,
        night_schedule: bool = None,
        night_schedule_start: int = None,
        night_schedule_end: int = None,
    ):
        """
        Generate a occupancy schedule based on the geometric moving average of electricity demand.

        Args:
            obj (dict, optional): Dictionary containing building parameters.
            data (dict, optional): Dictionary containing input data.
            results (dict, optional): Dictionary containing results from previously generated time series.
            ts_type (str, optional): Time series type to generate. Defaults to Types.OCCUPANCY.
            lambda_geoma (float, optional): Smoothing factor for the geometric moving average.
            night_schedule (bool, optional): Whether to apply the nightly schedule adjustment.
            night_schedule_start (int, optional): Start hour for nightly schedule (e.g., 18 for 6 PM).
            night_schedule_end (int, optional): End hour for nightly schedule (e.g., 6 for 6 AM).

        Returns:
            dict: Dictionary containing:
                - "summary" (dict):
                - "timeseries" (pd.DataFrame):
        Raises:
            Exception: If required data is missing or invalid.

        """

        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
        )

        # Get input data
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, results, ts_type)

        # Compute temperature and energy demand
        occ_schedule = calculate_timeseries(processed_obj, processed_data)

        return self._format_output(occ_schedule, processed_data)

    def _get_input_data(
        self, obj: dict, data: dict, results: dict, method_type: str = Types.OCCUPANCY
    ) -> tuple[dict, dict]:
        """Process and validate input data for Occupancy Schedule detection via GeoMA.

        This function extracts required and optional parameters from the input dictionaries,
        applies default values where needed, performs data validation, and prepares the
        data for Occupancy Schedule detection via GeoMA.

        Args:
            obj (dict): Dictionary containing GeoMA'a occupancy detection parameters.
            data (dict): Dictionary containing input data such as the electricity demand timeseries.
            results (dict): Dictionary containing results from previously generated time series.
            method_type (str, optional): Method type to use for prefixing. Defaults to Types.OCCUPANCY.

        Returns:
            tuple: A tuple containing:
                - obj_out (dict): Processed object parameters with defaults applied.
                - data_out (dict): Processed data with required format for calculation.

        Notes:
            - Parameters can be specified with method-specific prefixes (e.g., "occupancy:lambda")
              which will take precedence over generic parameters (e.g., "lambda").
        """
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
            {f"{Types.OCCUPANCY}{SEP}{O.OCCUPANCY}": occ_schedule[O.OCCUPANCY]},
            index=data[Types.ELECTRICITY].index,
        )
        df.index.name = C.DATETIME

        return {"summary": summary, "timeseries": df}


def calculate_timeseries(obj: dict, data: dict) -> pd.DataFrame:
    """Calculate occupancy schedule using Geometric Moving Average (GeoMA) method."""
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
