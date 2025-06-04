import importlib
import pkgutil

# Import registry functions
from entise.core.registry import method_registry, get_methods_by_type
from entise.constants.ts_types import VALID_TYPES

# Import all methods to ensure they're registered
for _, modname, _ in pkgutil.walk_packages(__path__, __name__ + "."):
    importlib.import_module(modname)

# Create a dictionary of all methods organized by type
methods_by_type = {ts_type: get_methods_by_type(ts_type) for ts_type in VALID_TYPES}

# Expose the registry and methods_by_type for direct access
__all__ = ['method_registry', 'methods_by_type']
