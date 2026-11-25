"""The interface for sinks that receive generated time series chunk by chunk.

EnTiSe ships no storage backend. A caller who wants to write results to files, a
database or another service while a large run is computed subclasses
:class:`TimeseriesStorage` and passes an instance to ``Generator.generate(storage=...)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class TimeseriesStorage(ABC):
    """Abstract sink for time series that ``Generator`` hands over chunk by chunk.

    ``Generator.generate(data, storage=sink)`` computes the objects in chunks of
    :attr:`chunk_size` and drives the lifecycle:

    1. :meth:`setup` once, before the first object is computed;
    2. :meth:`write_batch` once per chunk, in the order the objects were added;
    3. :meth:`finalize` once, after the last chunk.

    If anything raises after :meth:`setup` (an object with ``raise_on_error=True``,
    :meth:`write_batch` or :meth:`finalize`), the generator calls :meth:`close` once
    instead of :meth:`finalize` and re-raises the original error. Chunks that were
    written stay written: a sink that needs all-or-nothing runs handles that in
    :meth:`close`.

    The sink can also be used as a context manager around your own loop over
    ``Generator.iter_generate``: entering calls :meth:`setup`, leaving calls
    :meth:`finalize`, or :meth:`close` if the block raised. Don't combine the two:
    ``generate(storage=sink)`` sets the sink up itself.
    """

    #: Number of objects that ``Generator.generate(storage=...)`` computes and hands to
    #: :meth:`write_batch` per chunk. It bounds how many time series are in memory at once.
    chunk_size: int = 500

    @abstractmethod
    def setup(self) -> None:
        """Prepare the sink, for example by opening a connection or creating a folder."""

    @abstractmethod
    def write_batch(self, timeseries: Dict[Any, Dict[str, pd.DataFrame]]) -> None:
        """Write one chunk of results.

        Args:
            timeseries: ``{object id: {time series type: DataFrame}}`` for every object
                of the chunk, in input order. Each object is complete: all its types are
                in the same chunk. An object that failed with ``raise_on_error=False``
                maps to an empty dict; its error is in the summary.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Complete the run, for example by flushing buffers or committing."""

    def close(self) -> None:
        """Release resources after an error, without completing the run. Does nothing by default."""
        return None

    def __enter__(self) -> "TimeseriesStorage":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.close()
