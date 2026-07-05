import importlib
import pkgutil

from entise.constants.ts_types import VALID_TYPES

# Import registry functions
from entise.core.registry import get_methods_by_type, method_registry

# Import all methods to ensure they're registered. Modules whose basename
# starts with an underscore are treated as private (e.g., optional-dep
# accelerators like ``_R1C1_numba``); they are imported lazily by their
# public dispatchers so that missing optional deps do not break
# ``import entise.methods``.
for _, modname, _ in pkgutil.walk_packages(__path__, __name__ + "."):
    if any(part.startswith("_") for part in modname.split(".")):
        continue
    importlib.import_module(modname)

# Create a dictionary of all methods organized by type
methods_by_type = {ts_type: get_methods_by_type(ts_type) for ts_type in VALID_TYPES}

# Expose the registry and methods_by_type for direct access
__all__ = ["method_registry", "methods_by_type"]
