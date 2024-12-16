import logging
import pandas as pd

from src.core.base import TimeSeriesMethod
from src.constants import Columns as C, Keys as K, Objects as O, Types
from src.utils.decorators import supported_types
from src.core.registry import Meta

logger = logging.getLogger(__name__)


@supported_types(Types.HVAC)
class DependentHeating(TimeSeriesMethod, metaclass=Meta):
    required_keys = {O.WEATHER: str}
    required_timeseries = {
        O.WEATHER: {
            K.COLUMNS: {C.DATETIME: pd.Timestamp},
            K.DTYPE: pd.DataFrame
        },
        Types.ELECTRICITY: {
            K.COLUMNS: {C.LOAD: float},
            K.DTYPE: pd.DataFrame
        }
    }
    dependencies = [Types.ELECTRICITY]

    def generate(self, obj: dict, data: dict, ts_type: str, dependencies: dict = None, **kwargs) -> (dict, pd.DataFrame):
        """
        Generate the heating timeseries.

        Parameters:
        - obj (dict): Objects-specific metadata.
        - data (dict): Timeseries data.
        - dependencies (dict): Generated timeseries for dependencies.

        Returns:
        - (dict, pd.DataFrame): Metrics and the generated timeseries.
        """
        # Get the dependent timeseries
        electricity = dependencies.get(Types.ELECTRICITY, None)
        if electricity is None:
            logger.error("Electricity timeseries is missing.")
            raise ValueError("Electricity timeseries is missing.")

        # Prepare inputs
        obj = self.prepare_inputs(obj, data, ts_type)

        # Use dependencies to adjust the heating calculation
        heating_demand = electricity[C.LOAD] * 0.8

        summary = {
            "heating_demand": heating_demand.sum(),
        }
        df = pd.DataFrame({
            C.DATETIME: data[O.WEATHER][C.DATETIME],
            C.DEMAND: heating_demand,
        })

        return summary, df
