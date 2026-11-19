"""Shared pytest configuration."""

import pandas as pd


def pytest_configure(config):
    """Fail on deprecations that EnTiSe code triggers, on every supported pandas line.

    The filters match only warnings attributed to an ``entise`` module, so a
    deprecation inside a dependency does not fail the tests. ``Pandas4Warning``
    exists only from pandas 3 on, so its filter is added only there.
    """
    config.addinivalue_line("filterwarnings", "error::FutureWarning:entise")
    if hasattr(pd.errors, "Pandas4Warning"):
        config.addinivalue_line("filterwarnings", "error::pandas.errors.Pandas4Warning:entise")
