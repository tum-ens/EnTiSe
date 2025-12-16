import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.methods.multiple.file import FileLoader


@pytest.fixture
def sample_data():
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"value": [1, 2, 3]}, index=index)


def test_fileloader_success(sample_data):
    obj = {O.ID: "house1", O.FILE: "external_input"}
    data = {"external_input": sample_data}

    loader = FileLoader()
    result = loader.generate(obj, data)

    assert "timeseries" in result
    assert result["timeseries"].equals(sample_data)
    assert isinstance(result["timeseries"], pd.DataFrame)


def test_fileloader_missing_key():
    obj = {O.ID: "house1", O.FILE: "missing_key"}
    data = {}

    loader = FileLoader()
    with pytest.raises(ValueError, match="expected timeseries key 'missing_key'"):
        loader.generate(obj, data)


def test_fileloader_wrong_type():
    obj = {O.ID: "house1", O.FILE: "not_a_dataframe"}
    data = {"not_a_dataframe": [1, 2, 3]}

    loader = FileLoader()
    with pytest.raises(TypeError, match="Expected a DataFrame for key 'not_a_dataframe'"):
        loader.generate(obj, data)
