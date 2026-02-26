import json
import os
from typing import Any, Dict, Tuple

import pandas as pd

from entise import Generator


def _load_dir_into(
    data_dir: str, sink: Dict[str, Any], parse_dates: bool = True, warn_on_overwrite: bool = False
) -> None:
    """Load all supported files from a directory into the provided dict.

    Files are processed in sorted order for determinism. CSVs become DataFrames; JSON files become Python dicts.
    """
    if not os.path.isdir(data_dir):
        return
    for file in sorted(os.listdir(data_dir)):
        fp = os.path.join(data_dir, file)
        if not os.path.isfile(fp):
            continue
        name, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext == ".csv":
            if warn_on_overwrite and name in sink:
                print(f"Warning: overwriting key '{name}' with {fp}")
            sink[name] = pd.read_csv(fp, parse_dates=parse_dates)
        elif ext == ".json":
            if warn_on_overwrite and name in sink:
                print(f"Warning: overwriting key '{name}' with {fp}")
            with open(fp, "r", encoding="utf-8") as f:
                sink[name] = json.load(f)


def load_input(
    base_path: str = ".", load_common_data: bool = True, parse_dates: bool = True, warn_on_overwrite: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load example inputs from an example directory.

    Args:
        base_path: Path to an example folder (contains `objects.csv`, `data/`, optional `../common_data/`).
        load_common_data: Whether to also load files from `../common_data`.
        parse_dates: Whether to enable pandas' automatic date parsing for CSV files.
        warn_on_overwrite: Print a warning when a later file overwrites an existing key in the `data` dict.

    Returns:
        Tuple of `(objects_df, data_dict)` where `data_dict` maps file stems to loaded payloads.

    Raises:
        FileNotFoundError: If `objects.csv` is missing.
        ValueError: If `objects.csv` has no rows.
    """
    base_path = os.path.abspath(base_path)

    objects_path = os.path.join(base_path, "objects.csv")
    if not os.path.isfile(objects_path):
        raise FileNotFoundError(f"Missing objects.csv in {base_path}")
    objects = pd.read_csv(objects_path)
    if len(objects) == 0:
        raise ValueError(f"objects.csv in {base_path} contains no rows")

    data: Dict[str, Any] = {}

    # 1) Common first (lower precedence)
    if load_common_data:
        common_dir = os.path.join(base_path, "..", "common_data")
        _load_dir_into(common_dir, data, parse_dates=parse_dates, warn_on_overwrite=warn_on_overwrite)

    # 2) Example data last (overrides common on key collisions)
    data_dir = os.path.join(base_path, "data")
    _load_dir_into(data_dir, data, parse_dates=parse_dates, warn_on_overwrite=warn_on_overwrite)

    return objects, data


def run_simulation(
    objects: pd.DataFrame, data: Dict[str, Any], workers: int = 1
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the generator and return `(summary, df)`.

    Args:
        objects: Objects DataFrame (non-empty).
        data: Data dict as returned by `load_input`.
        workers: Degree of parallelism (>= 1).

    Returns:
        `(summary_df, results_dict)` from the generator.
    """
    gen = Generator()
    gen.add_objects(objects)
    return gen.generate(data, workers=workers)
