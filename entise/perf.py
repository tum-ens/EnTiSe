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

_VALID = {"auto", "numba", "none"}
_ACCELERATOR = os.environ.get("ENTISE_ACCELERATOR", "auto").lower()
if _ACCELERATOR not in _VALID:
    _ACCELERATOR = "auto"


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
    if _ACCELERATOR == "none":
        return "none"
    if _ACCELERATOR == "numba":
        if not _numba_available():
            raise ImportError(
                "ENTISE_ACCELERATOR=numba was requested but numba is not installed. "
                "Install with `pip install entise[numba]`, or set ENTISE_ACCELERATOR=auto "
                "to silently fall back to the numpy path."
            )
        return "numba"
    return "numba" if _numba_available() else "none"


def set_accelerator(mode: str) -> None:
    """Override the accelerator at runtime.

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
