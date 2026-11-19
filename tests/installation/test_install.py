"""Verify the built wheel in a clean environment outside the source tree."""

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]


def test_install(tmp_path):
    env_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", env_dir], check=True)
    scripts = env_dir / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    subprocess.run([python, "-m", "pip", "install", "uv"], check=True)

    dist_dir = tmp_path / "dist"
    subprocess.run([python, "-m", "uv", "build", "--wheel", "--out-dir", dist_dir], cwd=ROOT, check=True)
    (wheel,) = dist_dir.glob("entise-*.whl")
    subprocess.run([python, "-m", "uv", "pip", "install", "--python", python, wheel], check=True)

    check = dedent(
        """
        import sys
        from importlib.metadata import distribution
        from importlib.util import find_spec
        from pathlib import Path

        import entise
        from entise.core.registry import get_strategy_extra, list_strategies
        from entise.methods import method_registry, methods_by_type

        dist = distribution("entise")
        assert Path(entise.__file__).is_relative_to(Path(sys.prefix))
        assert entise.__version__ == dist.version != "unknown"
        base_requirements = {item.split(";", 1)[0] for item in dist.requires if ";" not in item}
        assert base_requirements == {"numpy>=2.0.0", "pandas<4,>=2.2.2", "joblib>=1.0.0", "tqdm>=4.0.0"}
        assert len(list_strategies()) == len(method_registry) == 16
        assert len(methods_by_type) == 10
        assert get_strategy_extra("districtheatingsim") is None
        assert get_strategy_extra("internalgainsoccupancy") is None
        assert method_registry["internalgainsoccupancy"].__name__ == "InternalOccupancy"
        provided_extras = set(dist.metadata.get_all("Provides-Extra"))
        assert "internalgainsoccupancy" not in provided_extras
        method_extras = {get_strategy_extra(name) for name in list_strategies()} - {None}
        assert len(method_extras) == 14
        assert method_extras <= provided_extras
        for package in ("pvlib", "scipy", "windpowerlib", "demandlib", "holidays", "pylpg", "pint", "matplotlib"):
            assert find_spec(package) is None, package
        """
    )
    subprocess.run([python, "-I", "-c", check], cwd=tmp_path, check=True)
