import os
import pandas as pd
import pytest

from entise.core.generator import TimeSeriesGenerator

DATA_DIR = os.path.dirname(__file__)

@pytest.fixture(scope="module")
def inputs():
    objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))
    data = {}
    data_folder = 'data'
    for file in os.listdir(os.path.join(DATA_DIR, data_folder)):
        if file.endswith('.csv'):
            name = file.split('.')[0]
            data[name] = pd.read_csv(os.path.join(os.path.join(DATA_DIR, data_folder, file)), parse_dates=True)

    return objects, data

def test_hvac_rc_all_objects(inputs):
    objects_df, shared_data = inputs

    for _, obj_row in objects_df.iterrows():
        gen = TimeSeriesGenerator()
        gen.add_objects(obj_row.to_dict())
        summary, df = gen.generate(shared_data, workers=1)
        print(summary)

        # assert "summary" in result
        # assert "timeseries" in result
        # ts = result["timeseries"]
        #
        # # Basic sanity checks
        # assert isinstance(ts, pd.DataFrame)
        # assert C.TEMP_IN in ts
        # assert len(ts) == 24 or len(ts) > 0  # depending on your input data duration
