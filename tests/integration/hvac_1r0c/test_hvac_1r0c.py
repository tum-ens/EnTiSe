import os

import pandas as pd
import pytest

from entise import Generator
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O

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


def test_hvac_1r0c_all_objects(inputs):
    objects_df, shared_data = inputs

    for _, obj_row in objects_df.iterrows():
        gen = Generator()
        gen.add_objects(obj_row.to_dict())
        summary, ts = gen.generate(shared_data, workers=1)

        # Basic sanity checks
        assert isinstance(summary, pd.DataFrame)
        assert isinstance(ts, dict)


@pytest.mark.parametrize("timestamped", [False, True], ids=["scalar", "timestamped"])
def test_hvac_1r0c_ventilation_activation(timestamped):
    """The ventilation flag controls heat loss from either input form."""
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    data = {O.WEATHER: pd.DataFrame({C.DATETIME: timestamps, C.TEMP_AIR: 0.0})}
    obj = {
        Types.HVAC: "1R0C",
        O.WEATHER: O.WEATHER,
        O.TIMEZONE: "UTC",
        O.RESISTANCE: 2.0,
        O.TEMP_INIT: 22.0,
        O.VENTILATION: 50.0,
    }
    if timestamped:
        column = "scheduled[W K-1]"
        data["ventilation_schedule"] = pd.DataFrame({C.DATETIME: timestamps, column: [25.0, 50.0, 75.0]})
        obj.update({O.VENTILATION: "ventilation_schedule", O.VENTILATION_COL: column})

    generator = Generator(raise_on_error=True)
    generator.add_objects(
        [
            {**obj, O.ID: "zero", O.VENTILATION: 0.0},
            {**obj, O.ID: "default"},
            {**obj, O.ID: "enabled", O.ACTIVE_VENTILATION: True},
            {**obj, O.ID: "disabled", O.ACTIVE_VENTILATION: False},
            {**obj, O.ID: "prefixed", O.ACTIVE_VENTILATION: True, f"{Types.HVAC}{SEP}{O.ACTIVE_VENTILATION}": False},
        ]
    )
    summary, timeseries = generator.generate(data, workers=1, show_progress=False)
    heating = summary[f"{Types.HEATING}{SEP}{C.DEMAND}[Wh]"]

    # At 0 °C outdoors, conduction accounts for 11 W in every hour.
    assert heating.loc["zero"] == pytest.approx(33.0)
    assert heating.loc["enabled"] == pytest.approx(3333.0)
    assert heating.loc["default"] == heating.loc["enabled"]
    pd.testing.assert_frame_equal(timeseries["default"][Types.HVAC], timeseries["enabled"][Types.HVAC])
    for case in ("disabled", "prefixed"):
        assert heating.loc[case] == heating.loc["zero"], case
        pd.testing.assert_frame_equal(timeseries[case][Types.HVAC], timeseries["zero"][Types.HVAC])
