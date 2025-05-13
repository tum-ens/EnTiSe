import logging
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

from entise.constants import Objects as O, Keys as K, VALID_TYPES
from entise.core.runner import RowExecutor
from entise.core.registry import get_strategy

class TimeSeriesGenerator:
    def __init__(self, logging_level=logging.WARNING, raise_on_error: bool = False):
        """
        Initializes the TimeSeriesGenerator.

        Args:
            logging_level (int): Logging level (e.g., logging.WARNING).
            raise_on_error (bool): Whether to raise exceptions on errors.

        Raises:
            ValueError: If `VALID_TYPES` is not defined or empty.
        """
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.objects = pd.DataFrame()
        self.raise_on_error = raise_on_error

    def add_objects(self, input_data: pd.DataFrame | dict | list[dict]):
        """
        Adds objects (metadata) to the generator.

        Args:
            input_data (dict, list, or pd.DataFrame): Input metadata for objects.

        Raises:
            ValueError: If any object is missing the required ID field.
            TypeError: If the input type is unsupported.
        """
        if isinstance(input_data, dict):
            input_data = [input_data]
        if isinstance(input_data, list):
            if not all(O.ID in obj for obj in input_data):
                raise ValueError(f"Each object in the list must have an {O.ID} field.")
            self.objects = pd.concat([self.objects, pd.DataFrame(input_data)], ignore_index=True)
        elif isinstance(input_data, pd.DataFrame):
            if O.ID not in input_data.columns:
                raise ValueError(f"The 'objects' DataFrame must have an {O.ID} column.")
            self.objects = pd.concat([self.objects, input_data], ignore_index=True)
        else:
            raise TypeError("Input must be a dictionary, list of dictionaries, or a pandas DataFrame.")

    def generate(self, data: Dict[str, pd.DataFrame], workers: int = -1) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generates time series data by processing objects in parallel or sequentially
        based on the number of workers specified.

        Args:
            data (dict): Dictionary containing input data used for processing.
                This dictionary includes the objects to be processed internally.
            workers (int, optional): Number of parallel workers to use for processing
                objects. The default is -1, which uses all available processors.

        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: A tuple containing the
            consolidated DataFrame of results and a dictionary of individual results
            per object.

        Raises:
            ValueError: If no objects are added to the processing pipeline before
            calling this method.
        """
        if self.objects.empty:
            raise ValueError("No objects have been added for processing.")

        object_params = self.objects.to_dict('records')

        if workers != 1:
            process = Parallel(n_jobs=workers, backend="loky")
            results = process(delayed(self._process_object)(obj, data) for obj in tqdm(object_params))
        else:
            results = [self._process_object(obj, data) for obj in tqdm(object_params)]

        return self._collect_results(results)

    def _process_object(self, obj: dict, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        obj_id = obj.get(O.ID, None)
        strategies = {k: v for k, v in obj.items() if k in VALID_TYPES and pd.notna(v)}
        static_inputs = {k: v for k, v in obj.items() if k not in strategies}

        executor = RowExecutor(static_inputs, data, strategies)
        main_outputs = executor.run_main_methods()

        summary = {}
        timeseries = {}
        for ts_type, result in main_outputs.items():
            summary.update(result.get(K.SUMMARY, {}))
            timeseries[ts_type] = result.get(K.TIMESERIES, pd.DataFrame())

        return {
            O.ID: obj_id,
            K.SUMMARY: summary,
            K.TIMESERIES: timeseries
        }

    def _collect_results(self, results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
        summaries = {}
        timeseries = {}

        for result in results:
            obj_id = result[O.ID]
            summaries[obj_id] = result[K.SUMMARY]
            timeseries[obj_id] = result[K.TIMESERIES]

        summary_df = pd.DataFrame.from_dict(summaries, orient="index")
        return summary_df, timeseries
