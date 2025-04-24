import os
import pandas as pd
from tqdm import tqdm


class ResultCollector:
    """
    A class to collect, manage, and export summary and timeseries results.

    This class processes results from timeseries generation methods, stores
    them in internal structures, and provides functionalities for exporting the
    results to various formats.
    """

    def __init__(self):
        """
        Initializes the ResultCollector instance.

        Attributes:
            summary_list (list): A list to store summary metrics for all objects.
            timeseries_dict (dict): A dictionary to store timeseries DataFrames keyed by object ID.
        """
        self.summary_list = []
        self.timeseries_dict = {}

    def collect(self, results: list) -> tuple[pd.DataFrame, dict]:
        """
        Collect results into internal structures.

        Args:
            results (list): A list of result dictionaries, each containing:
                - 'id' (str): Object ID.
                - 'summary' (dict): Summary metrics for the object.
                - 'timeseries' (pd.DataFrame): Timeseries DataFrame for the object.

        Returns:
            tuple:
                - pd.DataFrame: A DataFrame containing all collected summaries.
                - dict: A dictionary containing all collected timeseries keyed by object ID.

        Raises:
            ValueError: If a result dictionary is missing required keys.
        """
        for result in results:
            if "id" not in result or "summary" not in result or "timeseries" not in result:
                raise ValueError(f"Invalid result structure: {result}")
            self.summary_list.append(result["summary"])
            self.timeseries_dict[result["id"]] = result["timeseries"]

        return self.get_summary_df(), self.get_timeseries_dict()

    def get_summary_df(self) -> pd.DataFrame:
        """
        Retrieve the collected summaries as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all collected summaries.
        """
        return pd.DataFrame(self.summary_list)

    def get_timeseries_dict(self) -> dict:
        """
        Retrieve the collected timeseries as a dictionary.

        Returns:
            dict: Dictionary of timeseries DataFrames keyed by object ID.
        """
        return self.timeseries_dict

    def export_summary(self, filepath: str, file_format: str = "csv"):
        """
        Export the collected summary metrics to a file.

        Args:
            filepath (str): Path to the output file.
            file_format (str, optional): File format ('csv', 'json', 'excel'). Defaults to "csv".

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        summary_df = self.get_summary_df()

        match file_format:
            case "csv":
                summary_df.to_csv(filepath, index=False)
            case "json":
                summary_df.to_json(filepath, orient="records")
            case "excel":
                summary_df.to_excel(filepath, index=False)
            case _:
                raise ValueError(f"Unsupported file format: {file_format}")

    def export_timeseries(self, directory: str, file_format: str = "csv"):
        """
        Export collected timeseries to individual files.

        Each object's timeseries is exported as a separate file.

        Args:
            directory (str): Directory path where timeseries files will be saved.
            file_format (str, optional): File format ('csv', 'json', 'excel', 'feather'). Defaults to "csv".

        Raises:
            ValueError: If an unsupported file format is provided.
        """
        os.makedirs(directory, exist_ok=True)
        for obj_id, ts_df in tqdm(self.timeseries_dict.items(), desc="Exporting Timeseries"):
            match file_format:
                case "csv":
                    ts_df.to_csv(f"{directory}/{obj_id}_timeseries.csv", index=False)
                case "json":
                    ts_df.to_json(f"{directory}/{obj_id}_timeseries.json", orient="records")
                case "excel":
                    ts_df.to_excel(f"{directory}/{obj_id}_timeseries.xlsx", index=False)
                case "feather" | "ft":
                    ts_df.reset_index().to_feather(f"{directory}/{obj_id}_timeseries.ft")
                case _:
                    raise ValueError(f"Unsupported file format: {file_format}")

    def reset(self):
        """
        Reset the internal state of the ResultCollector.

        Clears all stored summaries and timeseries.
        """
        self.summary_list = []
        self.timeseries_dict = {}

    @staticmethod
    def get_available_outputs(methods: dict) -> dict:
        """
        Retrieve available outputs for all specified timeseries methods.

        Args:
            methods (dict): A dictionary of timeseries methods being used.

        Returns:
            dict: A dictionary mapping method names to their available outputs.
        """
        outputs = {}
        for name, method in methods.items():
            outputs[name] = method.available_outputs
        return outputs
