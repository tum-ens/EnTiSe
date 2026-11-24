"""A generation run reads each time table once and leaves the caller's data unchanged."""

import os
from unittest.mock import patch

import pandas as pd

from entise.constants import Objects as O
from entise.constants import Types
from entise.core import _time
from entise.core.generator import Generator
from entise.core.runner import Runner

HERE = os.path.dirname(__file__)


def _inputs() -> tuple[list[dict], dict]:
    folder = os.path.join(HERE, "hvac_1r1c")
    objects = pd.read_csv(os.path.join(folder, "objects.csv")).iloc[:3].to_dict("records")
    data = {name: pd.read_csv(os.path.join(folder, "data", f"{name}.csv")) for name in ("windows", "internal_gains")}
    data[O.WEATHER] = pd.read_csv(os.path.join(HERE, "common_data", "weather.csv")).iloc[:24]
    return objects, data


def test_each_time_table_is_parsed_once_per_run_and_reaches_every_method():
    objects, data = _inputs()
    original = {key: frame.copy() for key, frame in data.items()}
    seen = []
    runner_init = Runner.__init__

    def record(self, keys, run_data, strategies):
        seen.append(run_data.get(_time.RUN_TIME_KEY))
        runner_init(self, keys, run_data, strategies)

    generator = Generator(raise_on_error=True)
    generator.add_objects(objects)
    with (
        patch.object(_time, "parse_instants", wraps=_time.parse_instants) as parse,
        patch.object(Runner, "__init__", record),
    ):
        generator.generate(data, workers=1, show_progress=False)

    # weather and internal_gains carry timestamps; windows does not.
    assert parse.call_count == 2
    assert len(seen) == len(objects)
    assert all(isinstance(run, _time.RunTime) for run in seen)
    assert len({id(run) for run in seen}) == 1
    assert set(data) == set(original)
    for key, frame in data.items():
        pd.testing.assert_frame_equal(frame, original[key])


def test_hvac_returns_one_row_per_weather_row():
    objects, data = _inputs()

    generator = Generator(raise_on_error=True)
    generator.add_objects(objects[:1])
    _, timeseries = generator.generate(data, workers=1, show_progress=False)

    assert len(timeseries[objects[0][O.ID]][Types.HVAC]) == len(data[O.WEATHER])


def test_the_utc_option_reaches_the_methods():
    objects, data = _inputs()
    seen = []
    runner_init = Runner.__init__

    def record(self, keys, run_data, strategies):
        seen.append(run_data[_time.RUN_TIME_KEY].utc)
        runner_init(self, keys, run_data, strategies)

    generator = Generator(raise_on_error=True)
    generator.add_objects(objects[:1])
    with patch.object(Runner, "__init__", record):
        generator.generate(data, workers=1, show_progress=False)
        generator.generate(data, workers=1, show_progress=False, utc=True)

    assert seen == [False, True]
