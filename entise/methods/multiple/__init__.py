"""Methods shared by multiple time-series types."""

from functools import partial

from entise.methods._exports import list_exports, load_export

_EXPORTS = {"FileLoader": "file"}
__all__ = list(_EXPORTS)
__getattr__ = partial(load_export, __name__, _EXPORTS)
__dir__ = partial(list_exports, globals(), _EXPORTS)
