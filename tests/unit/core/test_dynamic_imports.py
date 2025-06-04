"""
Tests for the direct method import functionality.

This module tests the ability to import methods directly from the entise.methods
package and its subpackages.
"""

import importlib
import types


def test_import_methods_package():
    """Test importing the entise.methods package."""
    # Import the methods package
    import entise.methods

    # Check that the methods_by_type dictionary is available
    assert hasattr(entise.methods, "methods_by_type")
    assert isinstance(entise.methods.methods_by_type, dict)

    # Check that the method_registry is available
    assert hasattr(entise.methods, "method_registry")
    assert isinstance(entise.methods.method_registry, dict)

    # Check that the methods_by_type dictionary contains entries for each type
    from entise.constants.ts_types import VALID_TYPES

    for ts_type in VALID_TYPES:
        assert ts_type in entise.methods.methods_by_type


def test_import_specific_method_type():
    """Test importing a specific method type."""
    # Import the PV methods
    import entise.methods.pv

    # Check that the module is a module
    assert isinstance(entise.methods.pv, types.ModuleType)

    # Check that the module has the expected attributes
    assert hasattr(entise.methods.pv, "__all__")
    assert "PVLib" in entise.methods.pv.__all__


def test_access_method_directly():
    """Test accessing a method directly from a method type import."""
    # Import the PV methods
    import entise.methods.pv

    # Check that we can access the PVLib class directly
    assert hasattr(entise.methods.pv, "PVLib")

    # Check that we can instantiate the PVLib class
    pvlib = entise.methods.pv.PVLib()
    assert pvlib.name == "pvlib"
    assert pvlib.types == ["pv"]


def test_access_methods_by_type():
    """Test accessing methods by type from the main methods import."""
    # Import the methods package
    import entise.methods

    # Get the PV methods
    pv_methods = entise.methods.methods_by_type["pv"]

    # Check that the list contains at least one method
    assert len(pv_methods) > 0

    # Check that all methods in the list are for PV
    for method in pv_methods:
        assert "pv" in method.types


def test_import_all_method_types():
    """Test importing all method types that have corresponding directories."""
    # List of method types that have corresponding directories
    method_types = ["dhw", "electricity", "hvac", "mobility", "occupancy", "pv", "wind"]

    # Import each method type
    for ts_type in method_types:
        # Import the module
        module_name = f"entise.methods.{ts_type}"
        module = importlib.import_module(module_name)

        # Check that the module is a module
        assert isinstance(module, types.ModuleType)

        # Check that the module has the expected attributes
        assert hasattr(module, "__all__")

    # Special case for 'multiple' which should have the FileLoader class
    import entise.methods.multiple

    assert hasattr(entise.methods.multiple, "FileLoader")
    assert "FileLoader" in entise.methods.multiple.__all__
