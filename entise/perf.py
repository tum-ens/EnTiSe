"""Runtime accelerator selection for EnTiSe.

By default, methods that provide a numba-accelerated path use it if numba
is importable and fall back to the pure-numpy path otherwise. Users can
override the choice explicitly:

- environment variable  ``ENTISE_ACCELERATOR = auto | numba | none``
- Python API            ``entise.set_accelerator('numba')``

The three modes:

* ``auto`` (default): use numba if installed, otherwise numpy. No error.
* ``numba``: require numba. Raises ``ImportError`` at dispatch time if the
  package is not installed — useful when you want to fail loud instead
  of silently falling back.
* ``none``: force the numpy path regardless of whether numba is available.
  Useful for debugging, reproducibility, and side-by-side verification.

Only functions that explicitly opt in (currently the 1R1C HVAC solver)
consult this setting; everything else runs unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_VALID = {"auto", "numba", "none"}
_ACCELERATOR = os.environ.get("ENTISE_ACCELERATOR", "auto").lower()
if _ACCELERATOR not in _VALID:
    _ACCELERATOR = "auto"

_TASK_ACCELERATOR: ContextVar[str | None] = ContextVar("_TASK_ACCELERATOR", default=None)


def _get_accelerator_mode() -> str:
    """Return the requested mode without checking Numba availability."""
    return _TASK_ACCELERATOR.get() or _ACCELERATOR


@contextmanager
def _use_accelerator_mode(mode: str) -> Iterator[None]:
    """Apply a submitted mode for the duration of a worker task."""
    token = _TASK_ACCELERATOR.set(mode)
    try:
        yield
    finally:
        _TASK_ACCELERATOR.reset(token)


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


def get_accelerator() -> str:
    """Return the accelerator that will actually be used: ``'numba'`` or ``'none'``.

    Resolves the current mode against numba's install state:

    * ``auto`` → ``'numba'`` if numba is importable, else ``'none'``
    * ``numba`` → ``'numba'`` (raises ``ImportError`` here if numba is missing)
    * ``none`` → ``'none'``
    """
    mode = _get_accelerator_mode()
    if mode == "none":
        return "none"
    if mode == "numba":
        if not _numba_available():
            raise ImportError(
                "ENTISE_ACCELERATOR=numba was requested but numba is not installed. "
                "Install with `pip install entise[numba]`, or set ENTISE_ACCELERATOR=auto "
                "to silently fall back to the numpy path."
            )
        return "numba"
    return "numba" if _numba_available() else "none"


def set_accelerator(mode: str) -> None:
    """Set the default accelerator for this process at runtime.

    A call to ``Generator.generate()`` passes the current mode to its process
    workers, including workers reused from an earlier call.

    Args:
        mode: One of ``'auto'``, ``'numba'``, ``'none'``.

    Raises:
        ValueError: If ``mode`` is not one of the valid values.
    """
    global _ACCELERATOR
    lowered = mode.lower()
    if lowered not in _VALID:
        raise ValueError(f"Unknown accelerator mode {mode!r}; expected one of {sorted(_VALID)}")
    _ACCELERATOR = lowered
