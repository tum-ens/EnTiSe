"""Keep demandlib from changing the warning filters of the whole process."""

import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

_LOCK = threading.RLock()


@contextmanager
def contain_warning_filters() -> Iterator[None]:
    """Run demandlib code with the caller's warning filters.

    demandlib 0.2.2 calls ``warnings.simplefilter("error")`` while ``ElecSlp``
    builds a standard load profile. The filter would stay active for the rest of
    the process, so a later warning in any other method would raise. On pandas 3
    it also turns demandlib's own deprecation warnings into errors.

    Inside the context, that bare ``simplefilter("error")`` call is ignored and
    every other ``simplefilter`` call is passed on. The filters are restored on
    exit. While the context is active, it applies to every thread of the
    process. The lock keeps two threads from replacing ``simplefilter`` at the
    same time.
    """
    with _LOCK, warnings.catch_warnings():
        simplefilter = warnings.simplefilter

        def keep_caller_filters(action, *args, **kwargs):
            if action == "error" and not args and not kwargs:
                return None
            return simplefilter(action, *args, **kwargs)

        warnings.simplefilter = keep_caller_filters
        try:
            yield
        finally:
            warnings.simplefilter = simplefilter
