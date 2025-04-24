from joblib import Parallel, delayed
import logging
import pandas as pd
from tqdm import tqdm

from src.constants import Keys, Objects, VALID_TYPES
from src.core.dependency_resolver import DependencyResolver
from src.core.registry import get_method
from src.utils.result_collector import ResultCollector


class TimeSeriesGenerator:
    """
    A class to handle timeseries generation for multiple objects.

    This class supports sequential and parallel timeseries generation, resolves
    dependencies, and validates inputs.

    Attributes:
        logger (logging.Logger): Logger instance for logging messages.
        objects (pd.DataFrame): DataFrame containing metadata for the objects.
        raise_on_error (bool): Flag to raise exceptions on errors instead of warnings.
        dependency_resolver (DependencyResolver): Resolves the order of timeseries generation.
        result_collector (ResultCollector): Collects and manages generated results.
    """

    def __init__(self, logging_level=logging.WARNING, raise_on_error=False):
        """
        Initializes the TimeSeriesGenerator.

        Args:
            logging_level (int): Logging level (e.g., logging.WARNING).
            raise_on_error (bool): Whether to raise exceptions on errors.

        Raises:
            ValueError: If `VALID_TYPES` is not defined or empty.
        """
        if not VALID_TYPES:
            raise ValueError("VALID_TYPES is not defined or empty.")

        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.objects = pd.DataFrame()
        self.raise_on_error = raise_on_error
        self.dependency_resolver = DependencyResolver()
        self.result_collector = ResultCollector()

    def add_objects(self, input_data):
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
            if not all(Objects.ID in obj for obj in input_data):
                raise ValueError(f"Each object in the list must have an {Objects.ID} field.")
            self.objects = pd.concat([self.objects, pd.DataFrame(input_data)], ignore_index=True)
        elif isinstance(input_data, pd.DataFrame):
            if Objects.ID not in input_data.columns:
                raise ValueError(f"The 'objects' DataFrame must have an {Objects.ID} column.")
            self.objects = pd.concat([self.objects, input_data], ignore_index=True)
        else:
            raise TypeError("Input must be a dictionary, list of dictionaries, or a pandas DataFrame.")

    @staticmethod
    def get_method_requirements(method_name=None):
        """
        Retrieves input requirements for all methods or a specific method.

        Args:
            method_name (str, optional): Name of the specific method.

        Returns:
            dict: Requirements for all methods or the specified method.

        Raises:
            ValueError: If the specified method is not found in the registry.
        """
        from src.core.registry import method_registry

        if method_name:
            if method_name not in method_registry:
                raise ValueError(f"Method '{method_name}' not found in registry.")
            return {method_name: method_registry[method_name].get_requirements()}

        return {
            method: cls.get_requirements()
            for method, cls in method_registry.items()
        }

    def get_available_outputs(self) -> dict:
        """
        Retrieves available outputs for all timeseries methods.

        Returns:
            dict: Outputs grouped by summary and timeseries for each method.

        Note:
            This functionality is currently a placeholder for future development.
        """
        ts_methods = {
            k: get_method(v) for k, v in self.objects.iloc[0].items()
            if k in VALID_TYPES and pd.notna(v)
        }
        return self.result_collector.get_available_outputs(ts_methods)

    def generate(self, data: dict):
        """
        Generates timeseries data in parallel.

        Args:
            data (dict): Input data dictionary containing all timeseries.

        Returns:
            tuple: Summary DataFrame and a dictionary of generated timeseries.
        """
        return self._generate(data, parallel=True)

    def generate_sequential(self, data: dict):
        """
        Generates timeseries data sequentially.

        Args:
            data (dict): Input data dictionary containing all timeseries.

        Returns:
            tuple: Summary DataFrame and a dictionary of generated timeseries.
        """
        return self._generate(data, parallel=False)

    def _generate(self, data: dict, parallel: bool) -> (pd.DataFrame, dict):
        """
        Shared logic for sequential and parallel timeseries generation.

        Args:
            data (dict): Input data dictionary containing all timeseries.
            parallel (bool): Flag to enable parallel processing.

        Returns:
            tuple: A summary DataFrame and a dictionary of generated timeseries.

        Raises:
            ValueError: If no objects are available for processing.
        """
        if self.objects.empty:
            raise ValueError("No objects have been added for processing.")

        object_params = self.objects.to_dict('records')

        if parallel:
            process = Parallel(n_jobs=-1, backend="loky")
            results = process(delayed(self._process_object)(obj, data) for obj in tqdm(object_params))
        else:
            results = [self._process_object(obj, data) for obj in tqdm(object_params)]

        return self.result_collector.collect(results)

    def _process_object(self, obj: dict, data: dict) -> dict:
        """
        Processes a single object to generate its timeseries.

        Args:
            obj (dict): Metadata for the object.
            data (dict): Input data dictionary.

        Returns:
            dict: Generated summary and timeseries for the object.
        """
        obj_id = obj[Objects.ID]
        timeseries = {}
        summary_metrics = {}
        obj = self._pre_process_object(obj)
        ts_methods = {k: get_method(v) for k, v in obj.items() if k in VALID_TYPES and pd.notna(v)}
        sorted_ts_types = self.dependency_resolver.resolve(ts_methods)

        for ts_type in sorted_ts_types:
            method_class = ts_methods[ts_type]
            try:
                dependencies = {dep: timeseries[dep] for dep in method_class.dependencies if dep in timeseries}
                metrics, ts_df = method_class().generate(obj, data, ts_type=ts_type, dependencies=dependencies)
                summary_metrics.update(metrics)
                timeseries[ts_type] = ts_df
            except Exception as e:
                self._log_warning(f"Skipping '{ts_type}' for object ID {obj_id}: {e}")

        combined_timeseries = pd.concat(timeseries.values(), axis=1) if timeseries else pd.DataFrame()

        summary_metrics, combined_timeseries = self._post_process_object(obj, summary_metrics, combined_timeseries)

        return {Objects.ID: obj_id, Keys.SUMMARY: summary_metrics, Keys.TIMESERIES: combined_timeseries}

    def _pre_process_object(self, obj: dict) -> dict:
        """
        Placeholder for pre-processing hooks.

        Args:
            obj (dict): Object metadata.

        Returns:
            dict: Pre-processed object metadata.
        """
        return obj

    def _post_process_object(self, obj: dict, summary: dict, timeseries: pd.DataFrame) -> (dict, pd.DataFrame):
        """
        Placeholder for post-processing hooks.

        Args:
            obj (dict): Object metadata.
            summary (dict): Summary metrics for the object.
            timeseries (pd.DataFrame): Combined timeseries data.

        Returns:
            tuple: Updated summary metrics and timeseries data.
        """
        return summary, timeseries

    def _log_warning(self, msg: str):
        """
        Logs a warning or raises an exception based on configuration.

        Args:
            msg (str): The warning message.

        Raises:
            RuntimeError: If `raise_on_error` is True.
        """
        if self.raise_on_error:
            raise RuntimeError(msg)
        self.logger.warning(msg)
