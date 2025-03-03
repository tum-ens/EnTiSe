import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.utils.result_collector import ResultCollector


# 1. Testing the collect method
def test_collect_empty_results():
    collector = ResultCollector()
    summary_df, timeseries_dict = collector.collect([])
    assert summary_df.empty
    assert timeseries_dict == {}


def test_collect_duplicate_ids():
    collector = ResultCollector()
    results = [
        {"id": "object1", "summary": {"metric1": 1}, "timeseries": pd.DataFrame({"Time": [1], "Value": [10]})},
        {"id": "object1", "summary": {"metric2": 2}, "timeseries": pd.DataFrame({"Time": [2], "Value": [20]})},
    ]
    summary_df, timeseries_dict = collector.collect(results)
    assert len(summary_df) == 2  # Both summaries should be collected
    assert len(timeseries_dict) == 1  # Timeseries should overwrite for "object1"
    assert "object1" in timeseries_dict


def test_collect_incomplete_results():
    collector = ResultCollector()
    results = [
        {"id": "object1", "summary": {"metric1": 1}},  # Missing 'timeseries'
    ]
    with pytest.raises(ValueError, match="Invalid result structure"):
        collector.collect(results)


# 2. Testing edge cases for summary and timeseries retrieval
def test_get_summary_df_empty():
    collector = ResultCollector()
    df = collector.get_summary_df()
    assert df.empty


def test_get_timeseries_dict_empty():
    collector = ResultCollector()
    ts_dict = collector.get_timeseries_dict()
    assert ts_dict == {}


# 3. Testing exporting large datasets
@patch("src.utils.result_collector.os.makedirs")
@patch("src.utils.result_collector.pd.DataFrame.to_csv")
def test_export_large_summary(mock_to_csv, mock_makedirs):
    collector = ResultCollector()
    large_summary = [{"metric": i} for i in range(1000)]
    collector.summary_list = large_summary
    collector.export_summary("large_summary.csv")
    mock_to_csv.assert_called_once_with("large_summary.csv", index=False)
    assert len(collector.get_summary_df()) == 1000


@patch("src.utils.result_collector.os.makedirs")
@patch("src.utils.result_collector.pd.DataFrame.to_csv")
def test_export_large_timeseries(mock_to_csv, mock_makedirs):
    collector = ResultCollector()
    collector.timeseries_dict = {
        f"object_{i}": pd.DataFrame({"Time": [1, 2, 3], "Value": [i, i * 2, i * 3]}) for i in range(100)
    }
    collector.export_timeseries("output_dir")
    assert mock_to_csv.call_count == 100  # Verify all timeseries are exported


# 4. Error handling in exports
@patch("src.utils.result_collector.os.makedirs")
@patch("src.utils.result_collector.pd.DataFrame.to_csv", side_effect=Exception("File error"))
def test_export_summary_error(mock_to_csv, mock_makedirs):
    collector = ResultCollector()
    collector.summary_list = [{"metric1": 1}]
    with pytest.raises(Exception, match="File error"):
        collector.export_summary("error.csv")


@patch("src.utils.result_collector.os.makedirs")
@patch("src.utils.result_collector.pd.DataFrame.to_csv", side_effect=Exception("File error"))
def test_export_timeseries_error(mock_to_csv, mock_makedirs):
    collector = ResultCollector()
    collector.timeseries_dict = {"object1": pd.DataFrame({"Time": [1], "Value": [10]})}
    with pytest.raises(Exception, match="File error"):
        collector.export_timeseries("output_dir")
