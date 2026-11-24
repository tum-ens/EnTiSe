"""How a method resolves its declared keys from an object (#119)."""

import math

import pandas as pd
import pytest

from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method


class _Declared(Method):
    register = False
    types = [Types.HVAC]
    name = "_declared"
    required_keys = ["resistance[K W-1]"]
    optional_keys = {
        "area[m2]": 100.0,
        "active_heating": True,
        "power_heating[W]": 5000,
        "comment": None,
        "ventilation[W K-1]": None,
        O.TIMEZONE: None,
    }
    optional_data = ["ventilation[W K-1]"]

    def generate(self, obj=None, data=None, results=None, ts_type=Types.HVAC, **inputs):
        obj, data = self._process_kwargs(obj, data, ts_type, **inputs)
        return {"timeseries": pd.DataFrame(), "summary": self.get_relevant_objects(obj, ts_type)}


class _Listed(Method):
    register = False
    types = [Types.PV]
    name = "_listed"
    required_keys = ["latitude[degree]"]
    optional_keys = ["tilt[degree]", O.TIMEZONE]

    def generate(self, obj=None, data=None, results=None, ts_type=Types.PV, **inputs):
        raise NotImplementedError


def test_prefixed_key_comes_before_plain_key_before_default():
    obj = {"hvac:area[m2]": 80.0, "area[m2]": 90.0, "power_heating[W]": 3000}
    keys = _Declared.get_relevant_objects(obj, Types.HVAC)
    assert keys["area[m2]"] == 80.0
    assert keys["power_heating[W]"] == 3000
    assert keys["active_heating"] is True  # default


def test_another_type_prefix_is_ignored():
    keys = _Declared.get_relevant_objects({"pv:area[m2]": 80.0}, Types.HVAC)
    assert keys["area[m2]"] == 100.0


@pytest.mark.parametrize("missing", [None, math.nan])
def test_none_and_nan_count_as_missing(missing):
    obj = {"hvac:area[m2]": missing, "area[m2]": 90.0, "hvac:power_heating[W]": missing, "power_heating[W]": missing}
    keys = _Declared.get_relevant_objects(obj, Types.HVAC)
    assert keys["area[m2]"] == 90.0
    assert keys["power_heating[W]"] == 5000


@pytest.mark.parametrize("value", [False, 0])
def test_false_and_zero_are_values(value):
    obj = {"hvac:active_heating": value, "active_heating": True, "hvac:power_heating[W]": value}
    keys = _Declared.get_relevant_objects(obj, Types.HVAC)
    assert keys["active_heating"] is value
    assert keys["power_heating[W]"] is value


def test_keys_without_value_are_left_out_and_undeclared_keys_are_not_returned():
    obj = {"id": 1, "resistance[K W-1]": 0.01, "unknown": 1, "hvac:unknown": 2, "hvac": "5R1C"}
    keys = _Declared.get_relevant_objects(obj, Types.HVAC)
    assert "comment" not in keys
    assert "ventilation[W K-1]" not in keys
    assert not {"unknown", "hvac:unknown", "hvac"} & set(keys)
    assert keys["resistance[K W-1]"] == 0.01


def test_keys_come_in_declaration_order_after_the_site_keys():
    keys = _Declared.get_relevant_objects({"id": 1, "timezone": "Europe/Berlin", "resistance[K W-1]": 1}, Types.HVAC)
    assert list(keys) == ["id", "timezone", "resistance[K W-1]", "area[m2]", "active_heating", "power_heating[W]"]


def test_site_keys_are_read_without_prefix():
    obj = {"id": 7, "hvac:id": 8, "timezone": "Europe/Berlin", "hvac:timezone": "Asia/Tokyo"}
    keys = _Declared.get_relevant_objects(obj, Types.HVAC)
    assert keys["id"] == 7
    assert keys["timezone"] == "Europe/Berlin"


def test_a_missing_timezone_is_left_out():
    keys = _Declared.get_relevant_objects({"id": 7, "timezone": math.nan}, Types.HVAC)
    assert "timezone" not in keys


@pytest.mark.parametrize("reference", ["schedule|heating_on", "ventilation_table"])
def test_table_references_are_returned_as_given(reference):
    obj = {"hvac:active_heating": reference, "ventilation[W K-1]": reference}
    keys = _Declared.get_relevant_objects(obj, Types.HVAC)
    assert keys["active_heating"] == reference
    assert keys["ventilation[W K-1]"] == reference


def test_without_type_only_plain_keys_are_read():
    keys = _Declared.get_relevant_objects({"hvac:area[m2]": 80.0, "area[m2]": 90.0})
    assert keys["area[m2]"] == 90.0


def test_list_declarations_have_no_defaults():
    keys = _Listed.get_relevant_objects({"pv:latitude[degree]": 48.1, "tilt[degree]": 30}, Types.PV)
    assert keys == {"latitude[degree]": 48.1, "tilt[degree]": 30}


def test_several_scopes_take_the_value_given_under_any_of_them():
    obj = {"cooling:power_heating[W]": 2000, "power_heating[W]": 3000}
    keys = _Declared.get_relevant_objects(obj, (Types.HEATING, Types.COOLING))
    assert keys["power_heating[W]"] == 2000


def test_several_scopes_accept_the_same_value_twice():
    obj = {"heating:area[m2]": 80.0, "cooling:area[m2]": 80.0}
    assert _Declared.get_relevant_objects(obj, (Types.HEATING, Types.COOLING))["area[m2]"] == 80.0


def test_several_scopes_with_different_values_raise():
    obj = {"heating:area[m2]": 80.0, "cooling:area[m2]": 90.0}
    with pytest.raises(ValueError, match=r"heating:area\[m2\].*cooling:area\[m2\]"):
        _Declared.get_relevant_objects(obj, (Types.HEATING, Types.COOLING))


def test_key_defaults_cover_required_and_optional_keys():
    assert _Declared.key_defaults() == {"resistance[K W-1]": None, **_Declared.optional_keys}
    assert _Listed.key_defaults() == {"latitude[degree]": None, "tilt[degree]": None, O.TIMEZONE: None}


def test_requirements_list_the_keys_of_a_dict_declaration():
    requirements = _Declared.get_requirements()
    assert requirements["optional_keys"] == list(_Declared.optional_keys)
    assert requirements["required_keys"] == ["resistance[K W-1]"]


def test_direct_call_accepts_dict_declared_keys_by_python_name():
    summary = _Declared().generate(obj={"hvac:area[m2]": 80.0}, area=60.0, power_heating=0)["summary"]
    assert summary["area[m2]"] == 60.0
    assert summary["power_heating[W]"] == 0


def test_direct_call_routes_a_table_for_a_key_that_is_also_data():
    table = pd.DataFrame({"x": [1.0]})
    obj, data = _Declared()._process_kwargs({}, {}, Types.HVAC, ventilation=table)
    assert obj["ventilation[W K-1]"] == "ventilation[W K-1]"
    assert data["ventilation[W K-1]"] is table
