import os
import pandas as pd
from tqdm import tqdm


class ResultCollector:
    def __init__(self):
        self.summary_list = []
        self.timeseries_dict = {}

    def collect(self, results: list):
        """
        Collects results into internal structures.

        Parameters:
        - results (list): A list of result dictionaries, each containing:
            - 'id': Objects ID.
            - 'summary': Summary metrics for the object.
            - 'timeseries': Timeseries DataFrame for the object.

        Returns:
        - summary_df (pd.DataFrame): Collected summaries as a DataFrame.
        - timeseries_dict (dict): Collected timeseries keyed by object ID.
        """
        for result in results:
            if "id" not in result or "summary" not in result or "timeseries" not in result:
                raise ValueError(f"Invalid result structure: {result}")
            self.summary_list.append(result["summary"])
            self.timeseries_dict[result["id"]] = result["timeseries"]

        return self.get_summary_df(), self.get_timeseries_dict()

    def get_summary_df(self):
        """Returns the collected summaries as a DataFrame."""
        return pd.DataFrame(self.summary_list)

    def get_timeseries_dict(self):
        """Returns the collected timeseries as a dictionary."""
        return self.timeseries_dict

    def export_summary(self, filepath, file_format="csv"):
        """
        Exports the summary to a file.

        Parameters:
        - filepath (str): Path to the file.
        - file_format (str): File format ('csv', 'json', 'excel'). Default is 'csv'.
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

    def export_timeseries(self, directory: str, file_format="csv"):
        """
        Exports timeseries for each object to individual files.

        Parameters:
        - directory (str): Directory to save the timeseries files.
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
        """Resets the collector's state."""
        self.summary_list = []
        self.timeseries_dict = {}

    @staticmethod
    def get_available_outputs(methods: dict) -> dict:
        """
        Retrieve the available outputs for each method in use.

        Parameters:
        - methods (dict): A dictionary of timeseries methods being used.

        Returns:
        - dict: Outputs available for each method.
        """
        outputs = {}
        for name, method in methods.items():
            outputs[name] = method.available_outputs
        return outputs
