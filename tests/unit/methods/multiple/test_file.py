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
    assert isinstance(result["timeseries"], pd.DataFrame)
    # The values come back unchanged on the same instants (in UTC, as the object sets no time zone).
    assert list(result["timeseries"]["value"]) == list(sample_data["value"])
    assert result["timeseries"].index.equals(sample_data.index.as_unit("us"))


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


def test_fileloader_lists_its_types_in_a_stable_order():
    # The method reference pages list these types; a set would reorder them between builds.
    assert isinstance(FileLoader.types, list)
    assert FileLoader.types == sorted(FileLoader.types)
