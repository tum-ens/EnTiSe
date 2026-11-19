"""Importing EnTiSe must not add a filter that applies to every warning in the caller's process."""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

# Runs in a fresh interpreter, because the test session has usually imported these modules already.
CHECK = """
import importlib, pkgutil, warnings
import entise
changed = []
for module in pkgutil.walk_packages(entise.__path__, "entise."):
    before = list(warnings.filters)
    try:
        importlib.import_module(module.name)
    except ImportError:
        continue  # an optional dependency is missing
    # Dependencies may add narrow filters for their own messages; a blanket filter
    # (any message, every category, any module) silences or escalates everything.
    blanket = [f for f in warnings.filters if f not in before and f[1] is None and f[2] is Warning and f[3] is None]
    if blanket:
        changed.append((module.name, blanket))
print(changed)
assert not changed, changed
"""


def test_importing_entise_adds_no_blanket_warning_filter():
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))}

    result = subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
