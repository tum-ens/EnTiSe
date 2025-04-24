import pytest
import unittest
from unittest.mock import MagicMock
from src.core.base import TimeSeriesMethod
from src.utils.validation import Validator
from src.constants import Keys, SEP, Objects as O


class TestTimeSeriesMethod(unittest.TestCase):
    def setUp(self):
        # Create a mock subclass since TimeSeriesMethod is abstract
        class MockTimeSeriesMethod(TimeSeriesMethod):
            required_keys = {
                "param1": str,
                "param2": int
            }
            optional_keys = {
                "optional1": str,
                "optional2": bool
            }
            required_timeseries = {
                "param1": {}
            }
            optional_timeseries = {
                "optional1": {}
            }

            def generate(self, obj, data, ts_type, dependencies=None, **kwargs):
                pass

        self.method = MockTimeSeriesMethod()
        self.method.get_relevant_objects = MagicMock(wraps=self.method.get_relevant_objects)
        Validator.validate_object = MagicMock()
        Validator.validate_timeseries = MagicMock()

    def test_prepare_inputs_with_required_and_optional_keys(self):
        obj = {
            "param1": "ts1",
            "param2": 42,
            "optional1": "ts2",
            "optional2": True,
            "irrelevant_key": "should be ignored"
        }
        data = {
            "ts1": {"columnA": [], "columnB": []},
            "ts2": {"columnC": []}
        }
        ts_type = "test_ts"

        relevant_obj = self.method.prepare_inputs(obj, data, ts_type)

        # Ensure all required and optional keys are included
        self.assertIn("param1", relevant_obj)
        self.assertIn("param2", relevant_obj)
        self.assertIn("optional1", relevant_obj)
        self.assertIn("optional2", relevant_obj)
        self.assertNotIn("irrelevant_key", relevant_obj)

        # Validate method calls
        self.method.get_relevant_objects.assert_called_once_with(obj, ts_type)
        Validator.validate_object.assert_called_once()
        Validator.validate_timeseries.assert_called_once()

    def test_get_relevant_objects_filters_correctly(self):
        obj = {
            f"test_ts{SEP}param1": "ts1",
            "param2": 42,
            "optional1": "ts2",
            f"test_ts{SEP}optional2": False,
            "extra_key": "ignore me"
        }
        ts_type = "test_ts"

        relevant_obj = self.method.get_relevant_objects(obj, ts_type)

        expected_obj = {
            "param1": "ts1",  # Prefixed param1 should be used
            "param2": 42,  # Shared key should be used
            "optional1": "ts2",  # Optional should be included
            "optional2": False  # Prefixed optional should be used
        }

        self.assertDictEqual(relevant_obj, expected_obj)
        self.assertNotIn("extra_key", relevant_obj)  # Irrelevant key should be removed

    def test_get_relevant_objects_includes_optional_keys(self):
        obj = {
            f"test_ts{SEP}param1": 100,
            "optional1": 5.5,
            f"test_ts{SEP}optional2": True
        }
        ts_type = "test_ts"

        relevant_obj = self.method.get_relevant_objects(obj, ts_type)

        expected_obj = {
            "param1": 100,
            "optional1": 5.5,
            "optional2": True
        }

        self.assertDictEqual(relevant_obj, expected_obj)


if __name__ == "__main__":
    pytest.main()
