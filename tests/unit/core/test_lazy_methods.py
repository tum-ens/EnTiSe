"""Fresh-process checks for strategy discovery and optional imports."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[3]

BUILTINS = (
    "internalgainsoccupancy jordanvajen demandlib_electricity pylpg demandlib_heat districtheatingsim "
    "ruhnau 1r1c 5r1c 1r0c 7r2c file geoma pht pvlib wplib"
).split()

OPTIONAL_IMPORTS = set(
    "demandlib holidays matplotlib notebook numba openmeteo_requests pint pvlib pylpg "
    "requests_cache retry_requests scipy windpowerlib districtheatingsim".split()
)


def run_python(source):
    result = subprocess.run(
        [sys.executable, "-c", dedent(source)], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_import_and_strategy_names_load_no_methods_or_extras():
    run_python(
        f"""
        import sys
        import entise
        from entise.core.registry import list_strategies

        def check_imports():
            assert not {{name for name in sys.modules if name.startswith("entise.methods")}}
            assert not set(sys.modules).intersection({sorted(OPTIONAL_IMPORTS)!r})

        check_imports()
        assert list_strategies() == {BUILTINS!r}
        check_imports()
        """
    )


def test_1r1c_generation_only_loads_selected_method():
    run_python(
        f"""
        import sys
        import warnings
        import pandas as pd
        from entise import Generator, set_accelerator

        set_accelerator("none")
        # Exercise the solar path while leaving pvlib's own optional numba
        # acceleration independent of EnTiSe's solver setting.
        class NoNumba:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numba":
                    raise ModuleNotFoundError("No module named 'numba'", name="numba")

        sys.meta_path.insert(0, NoNumba())
        obj = pd.read_csv("examples/hvac_1r1c/objects.csv", nrows=1).iloc[0].to_dict()
        weather = pd.read_csv("examples/common_data/weather.csv", nrows=48)
        windows = pd.read_csv("examples/hvac_1r1c/data/windows.csv")
        gen = Generator(raise_on_error=True)
        gen.add_objects(obj)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*(?:spa to use numba|falling back to numpy).*")
            _, result = gen.generate(
                {{"weather": weather, obj["windows"]: windows}}, workers=1, show_progress=False
            )
        assert len(result[obj["id"]]["hvac"]) == len(weather)
        assert "entise.methods.hvac.R1C1" in sys.modules
        assert "pvlib" in sys.modules
        unexpected = {{
            name for name in sys.modules if name in {{
                "entise.methods.hvac.R1C0", "entise.methods.hvac.R5C1", "entise.methods.hvac.R7C2",
                "entise.methods.wind.wplib", "entise.methods.electricity.demandlib",
                "entise.methods.electricity.pylpg", "entise.methods.pv.pvlib",
            }}
        }} | set(sys.modules).intersection({sorted(OPTIONAL_IMPORTS - {"pvlib", "scipy"})!r})
        assert not unexpected, unexpected
        """
    )


@pytest.mark.parametrize("first", ["package", "registry", "submodule"])
def test_same_name_hvac_exports_preserve_class_identity(first):
    run_python(
        f"""
        import importlib
        import sys
        from entise.core.registry import get_strategy

        if {first!r} == "package":
            from entise.methods.hvac import R1C1 as first_class
        elif {first!r} == "registry":
            first_class = get_strategy("1R1C")
        else:
            from entise.methods.hvac.R1C1 import R1C1 as first_class

        from entise.methods.hvac import R1C1
        module = importlib.import_module("entise.methods.hvac.R1C1")
        assert first_class is R1C1 is module.R1C1 is get_strategy("1r1c")
        assert "entise.methods.hvac.R5C1" not in sys.modules
        """
    )


def test_selective_lookup_by_type_and_public_class_exports():
    run_python(
        """
        import sys
        from entise.core.registry import get_methods_by_type
        import entise.methods.pv

        assert "PVLib" in dir(entise.methods.pv)
        assert "entise.methods.pv.pvlib" not in sys.modules
        from entise.methods.pv import PVLib

        assert "pvlib" not in sys.modules
        assert [method.name for method in get_methods_by_type("pv")] == ["file", "pvlib"]
        assert "entise.methods.wind.wplib" not in sys.modules
        assert "entise.methods.hvac.R1C1" not in sys.modules

        from entise.constants import VALID_TYPES
        from entise.methods import method_registry, methods_by_type

        assert set(methods_by_type) == VALID_TYPES
        assert methods_by_type["pv"][-1] is method_registry["pvlib"] is PVLib
        assert len(method_registry) == 16
        from entise.core.registry import _BUILTIN_STRATEGIES

        for name, strategy in _BUILTIN_STRATEGIES.items():
            cls = method_registry[name]
            assert cls.__module__ == f"entise.methods.{strategy.module}"
            assert set(getattr(cls, "types", ())) == set(strategy.types)
        assert "pvlib" not in sys.modules
        """
    )


def test_catalog_covers_every_discoverable_strategy():
    run_python(
        """
        import entise.methods
        from entise.core.registry import import_all_methods, list_strategies, method_registry

        advertised = set(list_strategies())
        import_all_methods(entise.methods.__path__, entise.methods.__name__)
        assert advertised == set(method_registry)
        """
    )


@pytest.mark.parametrize("name", BUILTINS)
def test_strategy_lookup_imports_no_other_strategy_or_backend(name):
    run_python(
        f"""
        import sys
        from entise.core.registry import _BUILTIN_STRATEGIES, get_strategy

        cls = get_strategy({name!r})
        assert cls.name.lower() == {name!r}
        modules = {{f"entise.methods.{{entry.module}}" for entry in _BUILTIN_STRATEGIES.values()}}
        helpers = {{"entise.methods.auxiliary.internal.strategies"}} if "hvac" in cls.__module__ else set()
        assert modules.intersection(sys.modules) <= {{cls.__module__}} | helpers
        assert not set(sys.modules).intersection({sorted(OPTIONAL_IMPORTS)!r})
        """
    )


BLOCK_IMPORT = """
import importlib.abc
import sys

class BlockImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=missing)

sys.meta_path.insert(0, BlockImport())
"""


@pytest.mark.parametrize(
    ("blocked", "missing", "expected"),
    [
        ("windpowerlib", "windpowerlib", 'pip install "entise[wind-wplib]"'),
        ("windpowerlib", "broken_transitive_dependency", "No module named 'windpowerlib'"),
    ],
)
def test_missing_wind_dependency_through_direct_and_generator_calls(blocked, missing, expected):
    run_python(
        f"blocked = {blocked!r}\nmissing = {missing!r}\n"
        + BLOCK_IMPORT
        + dedent(
            f"""
            import pandas as pd
            from entise import Generator
            from entise.methods.wind import WPLib

            objects = pd.read_csv("examples/wind_wplib/objects.csv", nrows=1)
            weather = pd.read_csv("examples/common_data/weather.csv", nrows=48)
            obj = objects.iloc[0].to_dict()

            def run_generator(raise_on_error):
                gen = Generator(raise_on_error=raise_on_error)
                gen.add_objects(obj)
                return gen.generate({{"weather": weather}}, workers=1, show_progress=False)

            for call in (
                lambda: WPLib().generate(obj=obj, data={{"weather": weather}}),
                lambda: run_generator(True),
            ):
                try:
                    call()
                except ImportError as exc:
                    assert {expected!r} in str(exc), str(exc)
                    if missing != blocked:
                        assert isinstance(exc, ModuleNotFoundError) and exc.name == missing
                else:
                    raise AssertionError("missing dependency did not raise")

            summary, _ = run_generator(False)
            assert {expected!r} in summary.loc[obj["id"], "error"]
            """
        )
    )


@pytest.mark.parametrize(
    ("name", "example"),
    [("1r0c", "hvac_1r0c"), ("1r1c", "hvac_1r1c"), ("5r1c", "hvac_5r1c"), ("7r2c", "hvac_7r2c")],
)
def test_rc_methods_report_their_own_extra_with_solar_disabled(name, example):
    run_python(
        "blocked = missing = 'pvlib'\n"
        + BLOCK_IMPORT
        + dedent(
            f"""
            import pandas as pd
            from entise import Generator
            from entise.constants import Objects as O

            obj = pd.read_csv("examples/{example}/objects.csv", nrows=1).iloc[0].to_dict()
            obj[O.ACTIVE_GAINS_SOLAR] = False
            data = {{"weather": pd.read_csv("examples/common_data/weather.csv", nrows=48)}}
            generator = Generator()
            generator.add_objects(obj)
            summary, _ = generator.generate(data, workers=1, show_progress=False)
            assert 'pip install "entise[hvac-{name}]"' in summary.loc[obj["id"], "error"]
            """
        )
    )


def test_missing_weather_reports_install_command():
    run_python(
        "blocked = missing = 'pvlib'\n"
        + BLOCK_IMPORT
        + dedent(
            """
            try:
                from entise.services.weather import OpenMeteoProvider
            except ImportError as exc:
                assert 'pip install "entise[weather]"' in str(exc), str(exc)
            else:
                raise AssertionError("missing weather dependency did not raise")
            """
        )
    )


def test_missing_lpg_reports_install_command_through_generator():
    run_python(
        "blocked = missing = 'pylpg'\n"
        + BLOCK_IMPORT
        + dedent(
            """
            import pandas as pd
            from entise import Generator
            from entise.methods.electricity import PyLPG

            obj = pd.read_csv("examples/electricity_pylpg/objects.csv", nrows=1).iloc[0].to_dict()
            data = {"weather": pd.read_csv("examples/common_data/weather.csv", nrows=48)}
            install_command = 'pip install "entise[electricity-pylpg]"'

            try:
                PyLPG().generate(obj=obj, data=data)
            except ImportError as exc:
                assert install_command in str(exc), str(exc)
            else:
                raise AssertionError("missing LPG dependency did not raise")

            generator = Generator()
            generator.add_objects(obj)
            summary, _ = generator.generate(data, workers=1, show_progress=False)
            assert install_command in summary.loc[obj["id"], "error"]
            """
        )
    )
