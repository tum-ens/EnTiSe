import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from joblib import parallel as joblib_parallel
from tqdm import tqdm

from entise.constants import VALID_TYPES
from entise.constants import Keys as K
from entise.constants import Objects as O
from entise.core.runner import Runner


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.

    This avoids dependency on tqdm.contrib and works with joblib's internal
    BatchCompletionCallBack so the bar updates on task completion.
    """

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
        try:
            tqdm_object.close()
        except Exception:
            pass


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
        storage: Any = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generates time series data by processing objects in parallel or sequentially
        based on the number of workers specified. Supports batching to reduce
        scheduling and serialization overhead when running in parallel.

        If a ``storage`` sink is provided, results are streamed to it in chunks
        instead of being collected and returned: each chunk is computed, written to
        ``storage`` and then released, keeping memory bounded for very large runs. In
        that mode the returned time series dict is empty (the data lives in the sink)
        while the summary DataFrame is still returned in full.

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
            storage (TimeseriesStorage, optional): Sink implementing
                ``setup`` / ``write_batch`` / ``finalize``. When provided, the
                generated time series are streamed to it in chunks (of
                ``storage.chunk_size`` objects) and not returned. Defaults to None
                (legacy behaviour: collect and return).

        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: The consolidated summary
            DataFrame and a dictionary of per-object time series. When ``storage`` is
            used, the time series dictionary is empty.

        Raises:
            ValueError: If no objects are added to the processing pipeline before
            calling this method.
        """
        if self.objects.empty:
            raise ValueError("No objects have been added for processing.")

        object_params = self.objects.to_dict("records")

        if storage is not None:
            return self._generate_streaming(
                object_params,
                data,
                storage=storage,
                workers=workers,
                batch_size=batch_size,
                pre_dispatch=pre_dispatch,
                backend=backend,
                show_progress=show_progress,
            )

        results = self._execute(
            object_params, data, workers, batch_size, pre_dispatch, backend, show_progress
        )
        return self._collect_results(results)

    def _execute(
        self,
        object_params: List[Dict[str, Any]],
        data: Dict[str, pd.DataFrame],
        workers: int,
        batch_size: int | None,
        pre_dispatch: int | str,
        backend: str,
        show_progress: bool,
    ) -> List[Dict[str, Any]]:
        """Run the (sequential or parallel) processing of a list of objects.

        Returns the raw list of per-object result dicts (unconsolidated).
        """
        n = len(object_params)
        if n == 0:
            return []

        # Fast path: sequential
        if workers == 1:
            iterator = tqdm(object_params, disable=not show_progress, unit="obj")
            return [self._safe_process_object(obj, data) for obj in iterator]

        # Determine batch size (in objects per joblib-dispatched batch)
        if batch_size is None:
            # Split evenly across workers
            import math

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

        # Execute with a progress bar that advances by the number of objects completed
        if show_progress:
            with tqdm_joblib(tqdm(total=n, disable=not show_progress, unit="obj")):
                results = process(delayed(self._safe_process_object)(obj, data) for obj in object_params)
        else:
            results = process(delayed(self._safe_process_object)(obj, data) for obj in object_params)

        return results

    def _generate_streaming(
        self,
        object_params: List[Dict[str, Any]],
        data: Dict[str, pd.DataFrame],
        storage: Any,
        workers: int,
        batch_size: int | None,
        pre_dispatch: int | str,
        backend: str,
        show_progress: bool,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Compute objects in chunks, streaming each chunk to ``storage``.

        Keeps memory bounded (only one chunk of time series is held at a time) and
        lets the sink build its index once, in ``finalize()``. The chunk size is taken
        from the sink (``storage.chunk_size``).
        """
        chunk = max(1, int(getattr(storage, "chunk_size", 500)))
        n = len(object_params)
        summaries: Dict[Any, Any] = {}

        storage.setup()
        try:
            for start in range(0, n, chunk):
                sub = object_params[start : start + chunk]
                self.logger.info(
                    "Streaming chunk %d-%d of %d objects",
                    start + 1,
                    min(start + chunk, n),
                    n,
                )
                results = self._execute(
                    sub, data, workers, batch_size, pre_dispatch, backend, show_progress
                )

                chunk_ts: Dict[Any, Dict[str, pd.DataFrame]] = {}
                for result in results:
                    obj_id = result[O.ID]
                    summaries[obj_id] = result[K.SUMMARY]
                    chunk_ts[obj_id] = result[K.TIMESERIES]

                storage.write_batch(chunk_ts)
                del results, chunk_ts

            storage.finalize()
        except Exception:
            storage.close()
            raise

        summary_df = pd.DataFrame.from_dict(summaries, orient="index")
        return summary_df, {}

    def _process_object(self, obj: dict, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        obj_id = obj.get(O.ID, None)
        strategies = {k: v for k, v in obj.items() if k in VALID_TYPES and pd.notna(v)}
        keys = {k: v for k, v in obj.items() if k not in strategies}

        executor = Runner(keys, data, strategies)
        main_outputs = executor.run_methods()

        summary = {}
        timeseries = {}
        for ts_type, result in main_outputs.items():
            summary.update(result.get(K.SUMMARY, {}))
            timeseries[ts_type] = result.get(K.TIMESERIES, pd.DataFrame())

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

        summary_df = pd.DataFrame.from_dict(summaries, orient="index")
        return summary_df, timeseries
