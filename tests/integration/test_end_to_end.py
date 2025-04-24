import pytest
from src.core.timeseries_generator import TimeSeriesGenerator
from src.constants import Objects
from tests.utils.mock_methods import register_mock_methods
from tests.fixtures.object_fixtures import get_valid_object
from tests.fixtures.timeseries_fixtures import get_mock_timeseries_data


@pytest.fixture(scope="module", autouse=True)
def setup_mock_methods():
    """Register mock methods for the test session."""
    register_mock_methods()


def test_end_to_end_workflow():
    generator = TimeSeriesGenerator()

    # Add valid object
    valid_object = get_valid_object()
    generator.add_objects(valid_object)

    # Generate timeseries
    mock_timeseries = get_mock_timeseries_data()
    summary_df, timeseries_dict = generator.generate_sequential(mock_timeseries)

    # Assertions
    assert not summary_df.empty, "Summary DataFrame should not be empty."
    assert "metric1" in summary_df.columns, "Metric 'mock_metric1' should exist in the summary DataFrame."
    assert isinstance(timeseries_dict, dict), "Timeseries should be returned as a dictionary."
    assert valid_object[Objects.ID] in timeseries_dict, "Object ID should exist in the timeseries output."


def test_end_to_end_multiple_objects():
    generator = TimeSeriesGenerator()

    # Add multiple valid objects
    objects = [get_valid_object() for _ in range(2)]
    generator.add_objects(objects)

    # Generate timeseries
    mock_timeseries = get_mock_timeseries_data()
    summary_df, timeseries_dict = generator.generate_sequential(mock_timeseries)

    # Assertions
    assert not summary_df.empty, "Summary DataFrame should not be empty for multiple objects."
    assert len(summary_df) == 2, "Summary DataFrame should contain metrics for both objects."
    for obj in objects:
        obj_id = obj[Objects.ID]
        assert obj_id in timeseries_dict, f"Object ID {obj_id} should exist in the timeseries output."


if __name__ == "__main__":
    pytest.main()
