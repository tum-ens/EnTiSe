"""Methods for heating, ventilation, and air conditioning."""

import sys
from functools import partial
from types import ModuleType

from entise.methods._exports import list_exports, load_export

_EXPORTS = {"R1C1": "R1C1", "R5C1": "R5C1"}
__all__ = list(_EXPORTS)
__getattr__ = partial(load_export, __name__, _EXPORTS)
__dir__ = partial(list_exports, globals(), _EXPORTS)


class _HVACPackage(ModuleType):
    def __setattr__(self, name, value):
        # Importing a child normally binds its module to the parent. These two
        # names have historically exposed the class, including after direct imports.
        if name in _EXPORTS and isinstance(value, ModuleType) and value.__name__ == f"{__name__}.{name}":
            value = getattr(value, name)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _HVACPackage
