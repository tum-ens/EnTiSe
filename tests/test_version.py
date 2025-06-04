"""Test that the package can be imported and the version can be accessed."""

import entise


def test_version():
    """Test that the version is accessible."""
    assert entise.__version__ is not None
    assert entise.__version__ != "unknown"
