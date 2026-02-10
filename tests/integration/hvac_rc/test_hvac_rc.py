import os

import pandas as pd
import pytest

from entise.core.generator import Generator

DATA_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def inputs():
    objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))
    data = {}
    parent_dir = os.path.dirname(DATA_DIR)
    common_data_folder = os.path.join(parent_dir, "common_data")
    for file in os.listdir(common_data_folder):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            data[name] = pd.read_csv(os.path.join(os.path.join(common_data_folder, file)), parse_dates=True)
    data_folder = "data"
    for file in os.listdir(os.path.join(DATA_DIR, data_folder)):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            data[name] = pd.read_csv(os.path.join(os.path.join(DATA_DIR, data_folder, file)), parse_dates=True)

    return objects, data


def test_hvac_rc_all_objects(inputs):
    objects_df, shared_data = inputs

    for _, obj_row in objects_df.iterrows():
        gen = Generator()
        gen.add_objects(obj_row.to_dict())
        summary, ts = gen.generate(shared_data, workers=1)

        # Basic sanity checks
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(ts, dict)
