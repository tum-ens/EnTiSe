import warnings

import pytest

from entise.methods.utils._demandlib import contain_warning_filters


def test_code_inside_cannot_change_the_warning_filters():
    """demandlib 0.2.2 sets an error filter for the whole process while building a profile."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        before = list(warnings.filters)

        with contain_warning_filters():
            warnings.simplefilter("error")
            warnings.warn("deprecated inside demandlib", DeprecationWarning, stacklevel=2)

        assert list(warnings.filters) == before
    assert [warning.category for warning in caught] == [DeprecationWarning]


def test_warning_functions_are_restored_after_an_error():
    simplefilter = warnings.simplefilter

    with pytest.raises(RuntimeError), contain_warning_filters():
        raise RuntimeError("engine failed")

    assert warnings.simplefilter is simplefilter


def test_other_filter_changes_inside_still_apply():
    """Libraries such as pandas suppress their own warnings inside the context."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        with contain_warning_filters():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.warn("suppressed by a library", RuntimeWarning, stacklevel=2)

    assert caught == []


def test_nested_use_does_not_block():
    with contain_warning_filters(), contain_warning_filters():
        warnings.simplefilter("error")
