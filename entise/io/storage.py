"""Optional database storage backends for streaming EnTiSe time series output.

This module lets generated time series be written directly to a database *during*
generation, instead of being returned to the caller and held in memory. It is kept
intentionally generic: all domain-specific knowledge (which output columns map to
which series names, the target schema/table, units, etc.) is injected by the consuming
application via :class:`StorageConfig`, so EnTiSe itself stays domain-agnostic.

The PostgreSQL/TimescaleDB backend follows three design principles:

1. **Streaming** -- the generator hands over one chunk of results at a time; each chunk
   is written and released before the next is produced, so neither the generator nor
   the uploader ever holds the full result set in memory.
2. **COPY** -- each chunk is loaded with PostgreSQL ``COPY ... FROM STDIN`` (the fastest
   bulk load path), streamed client-side so it also works against remote/containerised
   databases. COPY runs per chunk (rather than one giant COPY for the whole run) to
   avoid a single huge transaction with long locks and unbounded WAL growth.
3. **Configurable index maintenance** -- ``index_strategy`` selects how the target index
   is kept current. ``"keep"`` creates it once, under a table-scoped advisory lock, and
   lets COPY maintain it incrementally, so any number of processes may stream into one
   shared table concurrently. ``"once"`` instead drops the index before an exclusive bulk
   load and rebuilds it a single time in :meth:`finalize`, trading concurrency for a
   faster one-shot build.
"""

from __future__ import annotations

import io
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)


@dataclass
class SeriesSpec:
    """Mapping from one EnTiSe output column to one stored database series.

    Attributes:
        column: Column name in the EnTiSe per-object time series DataFrame
            (e.g. ``"indoor_temperature[C]"``).
        name: Series name written to the metadata table
            (e.g. ``"ro_heat_indoor_temperature"``).
        description: Human-readable description written to the metadata table.
        type: Series type written to the metadata table (e.g. ``"synthetic"``).
        unit: Unit written to the metadata table (e.g. ``"degC"``).
    """

    column: str
    name: str
    description: str = ""
    type: str = ""
    unit: str = ""


@dataclass
class StorageConfig:
    """Configuration describing how and where to store generated time series.

    All domain-specific knowledge lives here so the storage backend stays generic.

    Attributes:
        schema: Target database schema.
        series: Sequence of :class:`SeriesSpec` describing the columns to store.
        ts_type: EnTiSe time series type to read from each object's result
            (e.g. ``"hvac"``). The matching DataFrame is looked up as
            ``timeseries[obj_id][ts_type]``.
        source: Value written to the metadata ``source`` column.
        data_table: Name of the time series data table.
        metadata_table: Name of the metadata table.
        index_name: Name of the data-table index on ``(ts_metadata_id, time)``.
        index_strategy: How that index is maintained -- ``"keep"`` (created once and left
            in place, safe for concurrent writers to a shared table) or ``"once"`` (dropped
            for an exclusive bulk load and rebuilt in :meth:`finalize`). See the module
            docstring for the trade-off.
        stream_chunk_size: Number of objects EnTiSe computes and writes per chunk when
            streaming to this sink. Bounds peak memory usage.
        synchronous_commit_off: If True, disable ``synchronous_commit`` during the COPY
            transaction for faster (slightly less durable) loads.
    """

    schema: str
    series: Sequence[SeriesSpec]
    ts_type: str
    source: str = "entise"
    data_table: str = "entise_ts_data"
    metadata_table: str = "entise_ts_metadata"
    index_name: str = "entise_ts_data_idx"
    index_strategy: str = "keep"
    stream_chunk_size: int = 500
    synchronous_commit_off: bool = True

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "StorageConfig":
        """Build a :class:`StorageConfig` from a plain dict (e.g. parsed YAML).

        The ``series`` entry must be a list of dicts compatible with
        :class:`SeriesSpec`. Keys that are not fields of this dataclass (e.g. caller
        flags such as ``store_timeseries``, ``stream_chunk_size`` or ``store_plz``) are
        ignored, so the full configuration block can be passed in as-is.
        """
        cfg = dict(cfg)
        series = [SeriesSpec(**s) for s in cfg.pop("series")]
        known = {f.name for f in fields(cls)} - {"series"}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        return cls(series=series, **kwargs)


class TimeseriesStorage(ABC):
    """Abstract sink for streamed EnTiSe time series.

    Lifecycle: :meth:`setup` once -> :meth:`write_batch` per chunk -> :meth:`finalize`
    once. On error paths, :meth:`close` releases resources without committing.
    """

    #: Number of objects EnTiSe computes and hands to :meth:`write_batch` per chunk.
    chunk_size: int = 500

    @abstractmethod
    def setup(self) -> None:
        """Prepare the sink (create tables, drop the load-time index, open buffers)."""

    @abstractmethod
    def write_batch(self, timeseries: Dict[Any, Dict[str, pd.DataFrame]]) -> None:
        """Persist one chunk of results.

        Args:
            timeseries: Mapping ``{obj_id: {ts_type: DataFrame}}`` for this chunk.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Flush remaining data, perform the bulk load and (re)build the index."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release resources without committing (used on error paths)."""

    def __enter__(self) -> "TimeseriesStorage":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.close()


class PostgresTimescaleStorage(TimeseriesStorage):
    """Stream EnTiSe output to a PostgreSQL/TimescaleDB table via COPY.

    See the module docstring for the streaming / COPY / index-once design.

    Args:
        engine: A SQLAlchemy engine connected to the target database.
        config: A :class:`StorageConfig` describing the target tables and the
            column-to-series mapping.
    """

    def __init__(self, engine: Any, config: StorageConfig):
        self.engine = engine
        self.config = config
        self.chunk_size = config.stream_chunk_size
        self._sa_text = text
        self._sa_MetaData = MetaData
        self._sa_Table = Table
        self._pg_insert = pg_insert

        self._is_setup = False
        self._total_rows = 0

    # ---- lifecycle ---------------------------------------------------------

    def setup(self) -> None:
        # Table and index creation is serialized across processes so concurrent runs
        # sharing a table cooperate rather than race on DDL.
        with self._ddl_lock():
            self._ensure_tables()
        if self.config.index_strategy == "once":
            self._drop_index()
        self._is_setup = True
        self._total_rows = 0
        logger.info("Storage ready; awaiting chunked COPY loads.")

    def write_batch(self, timeseries: Dict[Any, Dict[str, pd.DataFrame]]) -> None:
        if not timeseries:
            return
        if not self._is_setup:
            raise RuntimeError("write_batch() called before setup().")

        objectids = [str(obj_id) for obj_id in timeseries.keys()]
        meta_map = self._upsert_metadata(objectids)
        df = self._build_rows(timeseries, meta_map)
        if df.empty:
            return

        self._copy_dataframe(df)
        self._total_rows += len(df)

    def finalize(self) -> None:
        # Under "keep" the index already exists and COPY has maintained it; only "once"
        # rebuilds it here, after the index-free bulk load.
        if self.config.index_strategy == "once" and self._total_rows > 0:
            self._create_index()
        logger.info("Storage finalize complete; %s rows loaded.", f"{self._total_rows:,}")
        self._is_setup = False

    def close(self) -> None:
        # Per-chunk COPYs are already committed; nothing to roll back or clean up.
        self._is_setup = False

    # ---- DDL ---------------------------------------------------------------

    @contextmanager
    def _ddl_lock(self):
        """Serialize schema-object creation across processes writing to one table.

        Concurrent runs sharing a table must not issue overlapping DDL. The advisory lock
        is keyed to the target table, so the first arrival creates the objects while the
        others wait briefly and then observe them as already present; runs targeting
        different tables never block one another, and a single run pays no contention.
        """
        text = self._sa_text
        key = f"entise:ddl:{self.config.schema}.{self.config.data_table}"
        conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            conn.execute(text("SELECT pg_advisory_lock(hashtext(:key))"), {"key": key})
            yield
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(hashtext(:key))"), {"key": key})
            conn.close()

    def _ensure_tables(self) -> None:
        cfg = self.config
        text = self._sa_text
        # AUTOCOMMIT so a failure on one statement does not abort the rest.
        with self.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {cfg.schema};"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.metadata_table} (
                        id SERIAL PRIMARY KEY,
                        name text,
                        description text,
                        grid_id text,
                        type text,
                        unit text,
                        changelog integer,
                        objectid text,
                        source text
                    );
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS {cfg.metadata_table}_uniq
                    ON {cfg.schema}.{cfg.metadata_table} (name, objectid, source);
                    """
                )
            )

            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
            except Exception as exc:
                logger.warning("Could not ensure timescaledb extension (continuing): %s", exc)

            created = False
            try:
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.data_table} (
                            ts_metadata_id integer,
                            time timestamptz,
                            value double precision
                        )
                        WITH (
                            timescaledb.hypertable,
                            timescaledb.partition_column="time",
                            timescaledb.segmentby="ts_metadata_id"
                        );
                        """
                    )
                )
                created = True
            except Exception as exc:
                logger.warning("Hypertable creation failed, falling back to a plain table: %s", exc)

            if not created:
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {cfg.schema}.{cfg.data_table} (
                            ts_metadata_id integer,
                            time timestamptz,
                            value double precision
                        );
                        """
                    )
                )

            # Under "keep" the load-time index belongs to the table's steady state: it is
            # created once here and maintained incrementally by COPY, so concurrent writers
            # never drop or rebuild it. "once" leaves it out and manages it around the load.
            if cfg.index_strategy == "keep":
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {cfg.index_name} "
                        f"ON {cfg.schema}.{cfg.data_table} (ts_metadata_id, time DESC);"
                    )
                )

    def _drop_index(self) -> None:
        cfg = self.config
        text = self._sa_text
        with self.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                conn.execute(text(f"DROP INDEX IF EXISTS {cfg.schema}.{cfg.index_name};"))
            except Exception as exc:
                logger.debug("Could not drop index %s: %s", cfg.index_name, exc)

    def _create_index(self) -> None:
        cfg = self.config
        text = self._sa_text
        with self.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            try:
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {cfg.index_name} "
                        f"ON {cfg.schema}.{cfg.data_table} (ts_metadata_id, time DESC);"
                    )
                )
                conn.execute(text(f"ANALYZE {cfg.schema}.{cfg.data_table};"))
            except Exception as exc:
                logger.warning("Failed to (re)create index or ANALYZE: %s", exc)

    # ---- data --------------------------------------------------------------

    def _upsert_metadata(self, objectids: Sequence[str]) -> Dict[tuple, int]:
        """Upsert metadata for all (object, series) pairs and return a
        ``(name, objectid, source) -> id`` mapping using an ``ON CONFLICT`` upsert
        with ``RETURNING``.
        """
        cfg = self.config
        records = []
        for objectid in objectids:
            for spec in cfg.series:
                records.append(
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "type": spec.type,
                        "unit": spec.unit,
                        "changelog": 0,
                        "objectid": str(objectid),
                        "source": cfg.source,
                    }
                )

        if not records:
            return {}

        md = self._sa_MetaData()
        meta_table = self._sa_Table(
            cfg.metadata_table, md, schema=cfg.schema, autoload_with=self.engine
        )

        meta_map: Dict[tuple, int] = {}
        batch_size = 1000
        with self.engine.begin() as conn:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                insert_stmt = self._pg_insert(meta_table).values(batch)
                upsert_stmt = insert_stmt.on_conflict_do_update(
                    index_elements=[
                        meta_table.c.name,
                        meta_table.c.objectid,
                        meta_table.c.source,
                    ],
                    set_={
                        "unit": insert_stmt.excluded.unit,
                        "type": insert_stmt.excluded.type,
                        "description": insert_stmt.excluded.description,
                    },
                ).returning(
                    meta_table.c.id,
                    meta_table.c.name,
                    meta_table.c.objectid,
                    meta_table.c.source,
                )
                res = conn.execute(upsert_stmt)
                for row in res.mappings():
                    meta_map[(row["name"], row["objectid"], row["source"])] = row["id"]

        return meta_map

    def _build_rows(
        self, timeseries: Dict[Any, Dict[str, pd.DataFrame]], meta_map: Dict[tuple, int]
    ) -> pd.DataFrame:
        """Flatten one chunk of results into a ``(ts_metadata_id, time, value)`` frame."""
        cfg = self.config
        frames = []

        for obj_id, type_map in timeseries.items():
            ts_df = type_map.get(cfg.ts_type) if isinstance(type_map, dict) else None
            if ts_df is None or ts_df.empty:
                continue

            idx = ts_df.index
            for spec in cfg.series:
                ts_id = meta_map.get((spec.name, str(obj_id), cfg.source))
                if ts_id is None or spec.column not in ts_df.columns:
                    continue
                values = ts_df[spec.column].values
                if len(values) == 0:
                    continue
                frames.append(
                    pd.DataFrame({"ts_metadata_id": ts_id, "time": idx, "value": values})
                )

        if not frames:
            return pd.DataFrame(columns=["ts_metadata_id", "time", "value"])

        df = pd.concat(frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df[df["time"].notna()]
        return df

    def _copy_dataframe(self, df: pd.DataFrame) -> None:
        """Load one chunk into the data table via a single COPY from an in-memory CSV.

        The chunk is bounded by ``stream_chunk_size`` objects, so the buffer stays small
        regardless of the total run size.
        """
        cfg = self.config
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, date_format="%Y-%m-%dT%H:%M:%S%z")
        buf.seek(0)

        raw = self.engine.raw_connection()
        try:
            cur = raw.cursor()
            try:
                if cfg.synchronous_commit_off:
                    try:
                        cur.execute("SET LOCAL synchronous_commit = off;")
                    except Exception:
                        pass
                copy_sql = (
                    f"COPY {cfg.schema}.{cfg.data_table} (ts_metadata_id, time, value) "
                    f"FROM STDIN WITH (FORMAT csv)"
                )
                t0 = time.perf_counter()
                cur.copy_expert(copy_sql, buf)
                raw.commit()
                dt = time.perf_counter() - t0
                rows = len(df)
                rps = rows / dt if dt > 0 else rows
                logger.info("COPY chunk: %s rows in %.2fs (%.0f rows/s)", f"{rows:,}", dt, rps)
            finally:
                cur.close()
        finally:
            raw.close()
