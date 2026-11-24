"""The method reference generator reports modules it cannot import."""

import sys
from pathlib import Path

import pytest

DOCS_SOURCE = Path(__file__).resolve().parents[3] / "docs" / "source"


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(DOCS_SOURCE))
    import generate_methods_docs

    return generate_methods_docs


@pytest.fixture
def package_with_a_broken_module(tmp_path, monkeypatch):
    package = tmp_path / "docs_probe_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "broken.py").write_text("raise ImportError('missing optional dependency')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield package.name
    for name in [m for m in sys.modules if m.startswith(package.name)]:
        del sys.modules[name]


def test_an_unimportable_method_module_is_reported_to_the_caller(generator, package_with_a_broken_module):
    errors = []

    classes = generator.discover_method_classes(package_with_a_broken_module, on_error=errors.append)

    assert classes == []
    assert len(errors) == 1
    assert f"{package_with_a_broken_module}.broken" in errors[0]
    assert "missing optional dependency" in errors[0]
