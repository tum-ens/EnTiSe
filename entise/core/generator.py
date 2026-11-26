import logging
import math
import numbers
from collections.abc import Callable, Iterator
from collections.abc import Generator as GeneratorIterator
from contextlib import contextmanager
from os import getpid
from typing import Any, Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from joblib import parallel as joblib_parallel
from tqdm import tqdm

from entise.constants import SEP, VALID_TYPES
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.core._random import check_run_seed
from entise.core._time import RUN_TIME_KEY, UTC, RunTime
from entise.core.runner import Runner
from entise.core.storage import TimeseriesStorage
from entise.perf import _get_accelerator_mode, _use_accelerator_mode

logger = logging.getLogger(__name__)


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.

    This avoids dependency on tqdm.contrib and works with joblib's internal
    BatchCompletionCallBack so the bar updates on task completion. The bar is
    closed on exit.
    """
    try:
        with _joblib_progress(tqdm_object):
            yield tqdm_object
    finally:
        try:
            tqdm_object.close()
        except Exception:
            pass


@contextmanager
def _joblib_progress(tqdm_object):
    """Patch joblib to advance ``tqdm_object`` as batches complete, and leave the bar open."""

    class TqdmBatchCompletionCallback(joblib_parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            try:
                # Determine how many objects were processed in this completed batch.
                # Prefer the length of the returned batch results (args[0]) as it is
                # robust across joblib versions; fall back to self.batch_size.
                if args and hasattr(args[0], "__len__"):
                    n_update = len(args[0])
                else:
                    n_update = getattr(self, "batch_size", 1) or 1
                tqdm_object.update(n=int(n_update))
            except Exception:
                pass
            return super().__call__(*args, **kwargs)

    old_callback = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = old_callback


def _check_chunk_size(chunk_size: Any) -> int:
    """Return ``chunk_size`` as an int, or raise if it is not a whole number of at least 1."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, numbers.Integral) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a whole number of objects of at least 1, got {chunk_size!r}.")
    return int(chunk_size)


def _check_unique_ids(object_params: List[dict]) -> None:
    """Raise if two objects share an id; the results are keyed by id, so one would replace the other."""
    seen: set = set()
    repeated: List[Any] = []
    for obj in object_params:
        obj_id = obj.get(O.ID)
        if obj_id in seen and obj_id not in repeated:
            repeated.append(obj_id)
        seen.add(obj_id)
    if repeated:
        shown = ", ".join(map(repr, repeated[:10])) + (f" and {len(repeated) - 10} more" if len(repeated) > 10 else "")
        raise ValueError(f"Object ids must be unique; these repeat: {shown}.")


def _process_with_accelerator(
    process_object: Callable[[dict, Dict[str, pd.DataFrame]], Dict[str, Any]],
    obj: dict,
    data: Dict[str, pd.DataFrame],
    mode: str,
    caller_pid: int,
) -> Dict[str, Any]:
    """Use the submitting process mode for this task in a different process."""
    if getpid() == caller_pid:
        return process_object(obj, data)
    with _use_accelerator_mode(mode):
        return process_object(obj, data)


def _merge_summaries(obj_id: Any, summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge the summaries of an object's types into one mapping.

    A key that two or more types write, such as ``heating:demand[Wh]`` from an
    HVAC model and from a heat profile, is kept for each of them, prefixed with
    the type that wrote it (``hvac:heating:demand[Wh]``). Keys written by one
    type keep their names.
    """
    writers: Dict[str, List[str]] = {}
    for ts_type, values in summaries.items():
        for key in values:
            writers.setdefault(key, []).append(ts_type)
    shared = {key for key, types in writers.items() if len(types) > 1}
    if shared:
        example = min(shared)
        logger.warning(
            "Object '%s': the types %s all report %s; each value is kept with its type as a prefix, for example '%s'.",
            obj_id,
            sorted({ts_type for key in shared for ts_type in writers[key]}),
            sorted(shared),
            f"{writers[example][0]}{SEP}{example}",
        )
    merged: Dict[str, Any] = {}
    for ts_type, values in summaries.items():
        for key, value in values.items():
            merged[f"{ts_type}{SEP}{key}" if key in shared else key] = value
    return merged


class Generator:
    def __init__(self, logging_level=logging.WARNING, raise_on_error: bool = False):
        """
        Initializes the Generator.

        Args:
            logging_level (int): Logging level (e.g., logging.WARNING).
            raise_on_error (bool): Whether to raise exceptions on errors.

        Raises:
            ValueError: If `VALID_TYPES` is not defined or empty.
        """
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.objects = pd.DataFrame()
        self.raise_on_error = raise_on_error

    def add_objects(self, objects: pd.DataFrame | dict | list[dict]):
        """
        Adds objects (metadata) to the generator.

        Args:
            objects (dict, list, or pd.DataFrame): Input metadata for objects.

        Raises:
            ValueError: If any object is missing the required ID field.
            TypeError: If the input type is unsupported.
        """
        if isinstance(objects, dict):
            objects = [objects]
        if isinstance(objects, list):
            if not all(O.ID in obj for obj in objects):
                raise ValueError(f"Each object in the list must have an {O.ID} field.")
            self.objects = pd.concat([self.objects, pd.DataFrame(objects)], ignore_index=True)
        elif isinstance(objects, pd.DataFrame):
            if O.ID not in objects.columns:
                raise ValueError(f"The 'objects' DataFrame must have an {O.ID} column.")
            self.objects = pd.concat([self.objects, objects], ignore_index=True)
        else:
            raise TypeError("Input must be a dictionary, list of dictionaries, or a pandas DataFrame.")

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        workers: int = -1,
        batch_size: int | None = None,
        pre_dispatch: int | str = "2*n_jobs",
        backend: str = "loky",
        show_progress: bool = True,
        utc: bool = False,
        seed: int | None = None,
        storage: TimeseriesStorage | None = None,
    ) -> Tuple[pd.DataFrame, Dict[Any, Dict[str, pd.DataFrame]]]:
        """
        Generates time series data by processing objects in parallel or sequentially
        based on the number of workers specified. Process workers use the accelerator
        mode selected at the time of this call. Supports batching to reduce
        scheduling and serialization overhead when running in parallel.

        Args:
            data (dict): Dictionary containing input data used for processing.
                This dictionary includes the objects to be processed internally.
            workers (int, optional): Number of parallel workers to use for processing
                objects. The default is -1, which uses all available processors.
            batch_size (int | None, optional): Number of objects per dispatched job
                when running in parallel. If None, objects are split evenly across
                workers. Use smaller values (e.g., 8–64) for better load balancing
                when per-object runtimes vary. Defaults to None.
            pre_dispatch (int | str, optional): Controls how many jobs are dispatch
                ahead of time. Defaults to "2*n_jobs".
            backend (str, optional): Joblib backend to use ("loky" for processes,
                "threading" for threads). Defaults to "loky".
            show_progress (bool, optional): If True, display a progress bar. In
                parallel mode, the bar updates on batch completion. Defaults to True.
            utc (bool, optional): Methods index their outputs with ``output_index``,
                which uses each object's site time zone (the ``timezone`` key, UTC if
                not set). If True, those outputs are indexed in UTC instead. Methods
                that pass a table through, such as ``file``, keep its index.
                Defaults to False.
            seed (int | None, optional): Run seed for methods that draw random numbers.
                Each object and time series type draws from its own stream, derived
                from the run seed, the object's ``id`` and the type, so results do not
                depend on the order of the objects, the number of workers, the batch
                size or the process. An object's own ``seed`` takes precedence. If
                None, such methods draw different numbers in every run. Defaults to None.
            storage (TimeseriesStorage | None, optional): A sink that receives the time
                series instead of the caller. The objects are computed in chunks of
                ``storage.chunk_size`` (see :meth:`iter_generate`), and each chunk is
                passed to ``storage.write_batch`` and then released, so memory stays
                bounded for large runs. The generator calls ``storage.setup()`` before the
                first chunk and ``storage.finalize()`` after the last. If anything raises
                in between, it calls ``storage.close()`` instead and re-raises the error;
                chunks already written stay written. Defaults
                to None, which returns every time series.

        Returns:
            Tuple[pd.DataFrame, Dict[Any, Dict[str, pd.DataFrame]]]: The summary, with
            one row per object in the order the objects were added, and the time series
            as ``{object id: {time series type: DataFrame}}``. With ``storage``, the
            time series dict is empty, because the sink received them.

        Raises:
            ValueError: If no objects are added to the processing pipeline before
            calling this method, if object ids repeat, if ``seed`` is negative, or,
            with ``storage``, if ``storage.chunk_size`` is smaller than 1. These are
            checked before any object is computed.
            TypeError: If ``seed`` is not an integer.
        """
        if storage is not None:
            return self._generate_to_storage(
                data, storage, workers, batch_size, pre_dispatch, backend, show_progress, utc, seed
            )

        object_params, data = self._prepare(data, utc, seed)
        with self._progress(len(object_params), show_progress) as bar:
            results = self._execute(object_params, data, workers, batch_size, pre_dispatch, backend, bar)
        return self._collect_results(results)

    def iter_generate(
        self,
        data: Dict[str, pd.DataFrame],
        chunk_size: int = 500,
        workers: int = -1,
        batch_size: int | None = None,
        pre_dispatch: int | str = "2*n_jobs",
        backend: str = "loky",
        show_progress: bool = True,
        utc: bool = False,
        seed: int | None = None,
    ) -> Iterator[Tuple[pd.DataFrame, Dict[Any, Dict[str, pd.DataFrame]]]]:
        """Generate the objects in chunks and yield each chunk's results.

        Use it for runs too large to hold every time series in memory: only one chunk
        is computed at a time, and the next is computed when you ask for it. Each object
        is computed exactly as in :meth:`generate`, so the chunk size, the number of
        workers and the batch size don't change any value. Concatenating the chunk
        summaries with ``pd.concat`` gives the summary of :meth:`generate`.

        Each chunk holds up to ``chunk_size`` objects in the order they were added. An
        object is never split: all its time series types are in the same chunk. The
        arguments are checked when you call this method, before any object is computed.

        Args:
            data (dict): The input data, as for :meth:`generate`.
            chunk_size (int, optional): Number of objects per chunk. It bounds how many
                time series are held at once. Defaults to 500.
            workers, batch_size, pre_dispatch, backend, utc, seed: As for
                :meth:`generate`. ``batch_size`` applies within each chunk; with None,
                each chunk is split evenly across the workers.
            show_progress (bool, optional): If True, display one progress bar for all
                chunks. Defaults to True.

        Yields:
            Tuple[pd.DataFrame, Dict[Any, Dict[str, pd.DataFrame]]]: The summary of the
            chunk's objects and their time series as
            ``{object id: {time series type: DataFrame}}``.

        Raises:
            ValueError: If no objects are added, object ids repeat, ``chunk_size`` is
                smaller than 1, or ``seed`` is negative.
            TypeError: If ``seed`` is not an integer.
        """
        chunks = self._chunk_results(
            data, chunk_size, workers, batch_size, pre_dispatch, backend, show_progress, utc, seed
        )
        return self._iter_collected(chunks)

    def _iter_collected(
        self, chunks: GeneratorIterator[List[Dict[str, Any]], None, None]
    ) -> GeneratorIterator[Tuple[pd.DataFrame, Dict[Any, Dict[str, pd.DataFrame]]], None, None]:
        """Yield each chunk's summary and time series, holding no reference to a chunk the caller has moved past."""
        try:
            for results in chunks:
                collected = self._collect_results(results)
                del results
                yield collected
                del collected
        finally:
            chunks.close()

    def _chunk_results(
        self,
        data: Dict[str, pd.DataFrame],
        chunk_size: int,
        workers: int,
        batch_size: int | None,
        pre_dispatch: int | str,
        backend: str,
        show_progress: bool,
        utc: bool,
        seed: int | None,
    ) -> GeneratorIterator[List[Dict[str, Any]], None, None]:
        """Check the run now, and return an iterator over the per-object results of each chunk."""
        chunk_size = _check_chunk_size(chunk_size)
        object_params, data = self._prepare(data, utc, seed)
        return self._iter_chunks(
            object_params, data, chunk_size, workers, batch_size, pre_dispatch, backend, show_progress
        )

    def _iter_chunks(
        self,
        object_params: List[dict],
        data: Dict[str, Any],
        chunk_size: int,
        workers: int,
        batch_size: int | None,
        pre_dispatch: int | str,
        backend: str,
        show_progress: bool,
    ) -> GeneratorIterator[List[Dict[str, Any]], None, None]:
        with self._progress(len(object_params), show_progress) as bar:
            for start in range(0, len(object_params), chunk_size):
                chunk = object_params[start : start + chunk_size]
                yield self._execute(chunk, data, workers, batch_size, pre_dispatch, backend, bar)

    def _generate_to_storage(
        self,
        data: Dict[str, pd.DataFrame],
        storage: TimeseriesStorage,
        workers: int,
        batch_size: int | None,
        pre_dispatch: int | str,
        backend: str,
        show_progress: bool,
        utc: bool,
        seed: int | None,
    ) -> Tuple[pd.DataFrame, Dict[Any, Dict[str, pd.DataFrame]]]:
        """Feed the chunks of :meth:`iter_generate` into ``storage`` and return the summary."""
        chunk_size = getattr(storage, "chunk_size", TimeseriesStorage.chunk_size)
        chunks = self._chunk_results(
            data, chunk_size, workers, batch_size, pre_dispatch, backend, show_progress, utc, seed
        )
        # Keep each object's summary values, which are small, and build the frame once at
        # the end with the same call as generate().
        summaries: List[Dict[str, Any]] = []
        try:
            # As with a context manager, a sink whose setup fails is not closed.
            storage.setup()
            try:
                for results in chunks:
                    summaries.extend({O.ID: r[O.ID], K.SUMMARY: r[K.SUMMARY], K.TIMESERIES: {}} for r in results)
                    storage.write_batch({r[O.ID]: r[K.TIMESERIES] for r in results})
                    del results
                storage.finalize()
            except BaseException:
                try:
                    storage.close()
                except Exception:
                    self.logger.exception("Closing the time series storage after an error failed.")
                raise
        finally:
            chunks.close()
        summary_df, _ = self._collect_results(summaries)
        return summary_df, {}

    @contextmanager
    def _progress(self, total: int, show_progress: bool) -> Iterator[tqdm | None]:
        """One progress bar for a run, or None without one."""
        if not show_progress:
            yield None
            return
        bar = tqdm(total=total, unit="obj")
        try:
            yield bar
        finally:
            bar.close()

    def _prepare(self, data: Dict[str, pd.DataFrame], utc: bool, seed: int | None) -> Tuple[List[dict], Dict[str, Any]]:
        """Check the run and read every time table once; the caller's mapping is not changed."""
        if self.objects.empty:
            raise ValueError("No objects have been added for processing.")
        seed = check_run_seed(seed)
        object_params = self.objects.to_dict("records")
        _check_unique_ids(object_params)

        data = {**data, RUN_TIME_KEY: RunTime(data, utc=utc, seed=seed)}

        self._warn_missing_time_zones(object_params)
        return object_params, data

    def _execute(
        self,
        object_params: List[dict],
        data: Dict[str, Any],
        workers: int,
        batch_size: int | None,
        pre_dispatch: int | str,
        backend: str,
        bar: tqdm | None,
    ) -> List[Dict[str, Any]]:
        """Process the objects, sequentially or with joblib, and return their results in input order."""
        # Fast path: sequential
        if workers == 1:
            results = []
            for obj in object_params:
                results.append(self._safe_process_object(obj, data))
                if bar is not None:
                    bar.update(1)
            return results

        # Determine batch size (in objects per joblib-dispatched batch)
        n = len(object_params)
        if n == 0:
            return []

        if batch_size is None:
            # Split evenly across workers
            jobs = workers
            if jobs in (-1, 0):
                try:
                    from joblib.externals.loky import cpu_count

                    jobs = cpu_count()
                except Exception:
                    jobs = 1
            bs = max(1, math.ceil(n / max(1, jobs)))
        else:
            bs = max(1, int(batch_size))

        process = Parallel(
            n_jobs=workers,
            backend=backend,
            pre_dispatch=pre_dispatch,
            batch_size=bs,
            prefer="processes" if backend == "loky" else None,
        )

        mode = _get_accelerator_mode()
        caller_pid = getpid()
        tasks = (
            delayed(_process_with_accelerator)(self._safe_process_object, obj, data, mode, caller_pid)
            for obj in object_params
        )

        # Execute with a progress bar that advances by the number of objects completed
        if bar is None:
            return list(process(tasks))
        with _joblib_progress(bar):
            return list(process(tasks))

    def _warn_missing_time_zones(self, object_params: List[dict]) -> None:
        """Warn once per run about objects whose local calendar falls back to UTC."""
        missing = []
        for obj in object_params:
            value = obj.get(O.TIMEZONE)
            if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
                missing.append(str(obj.get(O.ID)))
        if not missing:
            return
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10 else "")
        self.logger.warning(
            "Objects without '%s' use %s as their local calendar and output time zone: %s. "
            "Set '%s' to an IANA name such as 'Europe/Berlin'.",
            O.TIMEZONE,
            UTC.key,
            shown,
            O.TIMEZONE,
        )

    def _process_object(self, obj: dict, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        obj_id = obj.get(O.ID, None)
        strategies = {k: v for k, v in obj.items() if k in VALID_TYPES and pd.notna(v)}
        keys = {k: v for k, v in obj.items() if k not in strategies}

        executor = Runner(keys, data, strategies)
        main_outputs = executor.run_methods()

        timeseries = {ts_type: result.get(K.TIMESERIES, pd.DataFrame()) for ts_type, result in main_outputs.items()}
        summaries = {ts_type: result.get(K.SUMMARY, {}) for ts_type, result in main_outputs.items()}
        summary = _merge_summaries(obj_id, summaries)

        return {O.ID: obj_id, K.SUMMARY: summary, K.TIMESERIES: timeseries}

    def _safe_process_object(self, obj: dict, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Wrapper around _process_object that captures exceptions per object.

        If raise_on_error is True, exceptions are propagated.
        Otherwise, returns a minimal result containing the error message in summary
        and empty time series for all types.
        """
        try:
            return self._process_object(obj, data)
        except Exception as exc:  # noqa: BLE001
            if self.raise_on_error:
                raise
            obj_id = obj.get(O.ID, None)
            return {O.ID: obj_id, K.SUMMARY: {"error": str(exc)}, K.TIMESERIES: {}}

    def _collect_results(
        self, results: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
        summaries = {}
        timeseries = {}

        for result in results:
            obj_id = result[O.ID]
            summaries[obj_id] = result[K.SUMMARY]
            timeseries[obj_id] = result[K.TIMESERIES]

        # One row per object, in input order, also for an object that reports nothing.
        summary_df = pd.DataFrame.from_dict(summaries, orient="index").reindex(list(summaries))
        return summary_df, timeseries
