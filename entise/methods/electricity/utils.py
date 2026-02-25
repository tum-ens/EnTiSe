"""
Helper utilities for the PyLPG method
"""

import contextlib
import logging
import os
import shutil
import tempfile
from typing import Dict, Optional

import pandas as pd

from entise.constants import Columns as C

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_fds():
    """Temporarily redirect OS-level stdout/stderr (fd 1 and 2) to os.devnull.
    Silences output from native extensions and child processes.

    Notes:
    - On Windows, low-level fd redirection can raise WinError 1 ("Incorrect function") in
      native/CLR code paths used by pylpg. To keep benchmarks stable, we disable
      fd-level silencing on Windows and simply yield (no-op). If needed, set the
      env var ENTISE_PYLPG_SILENCE_FDS=1 to force silencing (use with caution).
    """
    # No-op on Windows by default to avoid sporadic WinError 1 from native libs
    if os.name == "nt" and os.environ.get("ENTISE_PYLPG_SILENCE_FDS", "0") != "1":
        yield
        return

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            yield
        finally:
            # Restore original fds
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
    finally:
        os.close(devnull_fd)


def run_year_chunk(
    *,
    obj_id: str,
    year: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    households: int,
    occupants_per_household: int,
    energy_intensity_name: Optional[str],
) -> pd.DataFrame:
    """
    Run PyLPG for one chunk (minute resolution). Returns a DataFrame indexed by datetime.
    """
    try:
        import pylpg
        from pylpg.lpg_execution import execute_lpg_tsib
        from pylpg.lpgpythonbindings import EnergyIntensityType
    except Exception as e:
        raise ImportError("[pylpg] Requires the 'pyloadprofilegenerator' (pylpg) package.") from e

    tmp_base = None
    prev_cwd = os.getcwd()

    try:
        pylpg_root = os.path.dirname(pylpg.__file__)
        src_c1 = os.path.join(pylpg_root, "C1")

        tmp_base = os.path.join(tempfile.gettempdir(), f"pylpg_run_{os.getpid()}_{obj_id}_{year}")
        dst_c1 = os.path.join(tmp_base, "C1")

        if os.path.exists(dst_c1):
            shutil.rmtree(dst_c1, ignore_errors=True)
        shutil.copytree(src_c1, dst_c1)

        os.chdir(dst_c1)
        os.environ["PYLPG_HOME"] = dst_c1
    except Exception as e:
        logger.warning("[pylpg] Could not prepare isolated PyLPG dir (continuing): %s", e)

    # Energy intensity handling:
    energy_intensity = EnergyIntensityType.Random
    if energy_intensity_name:
        try:
            energy_intensity = getattr(EnergyIntensityType, str(energy_intensity_name))
        except Exception:
            logger.warning("[pylpg] Unknown energy_intensity='%s', using Random", energy_intensity_name)

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d %H:%M:%S")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Silence all outputs from the external simulator and any native libs it loads
        with _silence_fds():
            df = execute_lpg_tsib(
                year=int(year),
                number_of_households=int(households),
                number_of_people_per_household=int(occupants_per_household),
                startdate=start_s,
                enddate=end_s,
                energy_intensity=energy_intensity,
            )
    except Exception as e:
        raise RuntimeError(f"[pylpg] Execution failed for id={obj_id} year={year}: {e}") from e
    finally:
        try:
            os.chdir(prev_cwd)
        except Exception:
            pass
        if tmp_base and os.path.exists(tmp_base):
            shutil.rmtree(tmp_base, ignore_errors=True)

    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"[pylpg] Empty output for id={obj_id} year={year}")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    df = df.loc[(df.index >= start) & (df.index < end)].copy()
    df.index.name = C.DATETIME
    return df


def extract_hh_electricity_columns(df_min_all: pd.DataFrame, households: int) -> pd.DataFrame:
    """
    Find HH electricity columns in PyLPG output and return a DataFrame with those columns.
    If HH columns cannot be found, tries a 'house electricity' fallback.
    """
    df_min_all = df_min_all.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if df_min_all.empty:
        raise RuntimeError("[pylpg] Empty numeric output after coercion")

    hh_cols: Dict[int, str] = {}
    for col in df_min_all.columns:
        c = str(col)
        lc = c.lower()

        if "electric" not in lc:
            continue
        if "house" in lc:
            continue

        pos = lc.find("hh")
        if pos == -1:
            continue

        num = ""
        for ch in lc[pos + 2 :]:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            continue

        hh = int(num)
        if hh not in hh_cols:
            hh_cols[hh] = col

    if not hh_cols:
        house_like = [c for c in df_min_all.columns if "electric" in str(c).lower() and "house" in str(c).lower()]
        if not house_like:
            raise RuntimeError(
                "[pylpg] Could not find HH electricity columns or a House electricity column. "
                f"Columns (sample): {list(df_min_all.columns)[:40]}"
            )
        return df_min_all[[house_like[0]]].copy()

    keep_hhs = [hh for hh in sorted(hh_cols.keys()) if 1 <= hh <= households]
    if not keep_hhs:
        raise RuntimeError(f"[pylpg] No HH columns in range 1..{households}. Found: {sorted(hh_cols.keys())}")

    if len(keep_hhs) < households:
        logger.warning("[pylpg] Requested households=%s but only found HH columns: %s", households, keep_hhs)

    return df_min_all[[hh_cols[hh] for hh in keep_hhs]].copy()
