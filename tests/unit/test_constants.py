"""The constant classes are namespaces of plain strings that can be iterated and checked for membership."""

import pytest

from entise.constants import SEP, VALID_TYPES, Columns, Keys, Objects, Types

CLASSES = [Columns, Keys, Objects, Types]


def _declared(cls) -> list[tuple[str, str]]:
    """Name and value of each constant, in the order the class body defines them."""
    return [(name, value) for name, value in vars(cls).items() if not name.startswith("_")]


@pytest.mark.parametrize("cls", CLASSES)
def test_every_public_attribute_is_an_upper_case_plain_string(cls):
    for name, value in _declared(cls):
        assert name.isupper(), f"{cls.__name__}.{name}"
        assert type(value) is str, f"{cls.__name__}.{name}"


@pytest.mark.parametrize("cls", CLASSES)
def test_iteration_yields_the_values_in_definition_order(cls):
    assert list(cls) == [value for _, value in _declared(cls)]
    assert all(type(value) is str for value in cls)


@pytest.mark.parametrize("cls", CLASSES)
def test_len_counts_the_constants(cls):
    assert len(cls) == len(_declared(cls)) > 0


@pytest.mark.parametrize("cls", CLASSES)
def test_items_maps_names_to_values(cls):
    assert list(cls.items()) == _declared(cls)


@pytest.mark.parametrize("cls", CLASSES)
def test_values_are_unique_within_a_class(cls):
    values = [value for _, value in _declared(cls)]
    duplicates = sorted({value for value in values if values.count(value) > 1})
    assert not duplicates, f"{cls.__name__} repeats {duplicates}"


def test_membership_is_by_value():
    assert "dhw" in Types
    assert Types.DHW in Types
    assert "nope" not in Types
    assert "DHW" not in Types
    assert "id" in Objects and "id" in Columns
    assert "summary" in Keys
    assert None not in Types
    assert ["dhw"] not in Types


def test_attributes_keep_their_plain_string_values():
    assert type(Objects.ID) is str and Objects.ID == "id"
    assert repr(Types.DHW) == "'dhw'"
    assert f"{Types.DHW}{SEP}{Objects.SEED}" == "dhw:seed"


def test_attributes_built_from_other_attributes_hold_the_combined_string():
    assert Columns.TEMP_WATER_HOT == "hot_water_temperature[C]"
    assert Columns.DAY_OF_WEEK == "day_of_week"
    assert Objects.ACTIVE_HEATING == "active_heating"
    assert Objects.TEMP_WATER_COLD == "cold_water_temperature[C]"
    assert Keys.COLS_REQUIRED == "required_columns"
    assert Keys.KEYS_OPTIONAL == "optional_keys"


def test_valid_types_are_the_type_strings():
    assert VALID_TYPES == set("cooling dhw electricity heating hp hvac mobility occupancy pv wind".split())
    assert VALID_TYPES == set(Types)
    assert all(type(ts_type) is str for ts_type in VALID_TYPES)


@pytest.mark.parametrize("cls", CLASSES)
def test_constants_cannot_be_changed_added_or_removed(cls):
    name = next(iter(cls.items()))[0]
    with pytest.raises(AttributeError):
        setattr(cls, name, "changed")
    with pytest.raises(AttributeError):
        cls.NEW_CONSTANT = "new"
    with pytest.raises(AttributeError):
        delattr(cls, name)
    assert list(cls) == [value for _, value in _declared(cls)]


@pytest.mark.parametrize("cls", CLASSES)
def test_the_classes_are_not_instantiated(cls):
    with pytest.raises(TypeError):
        cls()
