import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """Keep the weather cache files (.cache.sqlite, .test_cache.sqlite) out of the repository.

    OpenMeteoProvider opens its requests cache at a path relative to the working directory.
    """
    monkeypatch.chdir(tmp_path)
