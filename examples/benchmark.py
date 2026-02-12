"""
Benchmark runner for EnTiSe example methods.

This script discovers example method folders under examples/ (any subfolder
containing an objects.csv) and benchmarks their generation performance using
the shared utilities in examples.utils. It does not import the examples'
runme.py files and performs no plotting by default.

Features:
- Select methods via --methods (comma-separated) or use 'all' to benchmark all discovered methods.
- Configure parallel workers via --workers and dataset size via --objects (both accept comma-separated lists).
- Repeat each configuration with --repeats and record per-run metrics.
- Outputs a timestamped CSV with columns:
  method, num_workers, num_objects, runtime_sec, iter_per_sec, speedup, status, error

Usage examples (from repository root):
- python examples\\benchmark.py
- python examples\\benchmark.py --methods pv_pvlib,hp_ruhnau --workers 1,2,4 --objects 50,100 --repeats 3
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is importable
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from examples.utils import load_input, run_simulation  # noqa: E402

EXAMPLES_DIR = THIS_DIR


def parse_int_list(csv_str: str) -> List[int]:
    return [int(x.strip()) for x in csv_str.split(",") if x.strip()]


def discover_methods(examples_dir: str) -> List[str]:
    methods: List[str] = []
    for name in os.listdir(examples_dir):
        path = os.path.join(examples_dir, name)
        if not os.path.isdir(path):
            continue
        if name.lower() in {"common_data", "__pycache__"}:
            continue
        if os.path.isfile(os.path.join(path, "objects.csv")):
            methods.append(name)
    methods.sort()
    return methods


def scale_objects(objects: pd.DataFrame, target_n: int) -> pd.DataFrame:
    n0 = len(objects)
    if n0 <= 0:
        raise ValueError("objects.csv contains no rows")
    if target_n <= n0:
        return objects.iloc[:target_n].reset_index(drop=True)
    reps = (target_n + n0 - 1) // n0
    out = pd.concat([objects] * reps, ignore_index=True)
    return out.iloc[:target_n].reset_index(drop=True)


def benchmark_one(
    base: str,
    method: str,
    cached: Tuple[pd.DataFrame, Dict[str, object]],
    workers: int,
    num_objects: int,
) -> Dict[str, object]:
    try:
        objects, data = cached
        objects_scaled = scale_objects(objects, num_objects)

        t0 = time.perf_counter()
        _summary, _df = run_simulation(objects_scaled, data, workers=workers)
        dt = time.perf_counter() - t0

        return {
            "method": method,
            "num_workers": workers,
            "num_objects": num_objects,
            "runtime_sec": dt,
            "iter_per_sec": (num_objects / dt) if dt > 0 else None,
            "status": "ok",
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "method": method,
            "num_workers": workers,
            "num_objects": num_objects,
            "runtime_sec": None,
            "iter_per_sec": None,
            "status": "failed",
            "error": str(e),
        }


def run_sweep(methods: List[str], workers: List[int], objects_list: List[int], repeats: int) -> pd.DataFrame:
    # Pre-load inputs once per method
    cache: Dict[str, Tuple[pd.DataFrame, Dict[str, object]]] = {}
    rows: List[Dict[str, object]] = []

    for method in methods:
        base_path = os.path.join(EXAMPLES_DIR, method)
        try:
            cache[method] = load_input(base_path, load_common_data=True)
        except Exception as e:  # noqa: BLE001
            # If loading fails, record a single error row per combination and skip running
            for nobj in objects_list:
                for w in workers:
                    for _ in range(repeats):
                        rows.append(
                            {
                                "method": method,
                                "num_workers": w,
                                "num_objects": nobj,
                                "runtime_sec": None,
                                "iter_per_sec": None,
                                "status": "failed",
                                "error": f"load_input: {e}",
                            }
                        )
            continue

        for nobj in objects_list:
            for w in workers:
                for r in range(repeats):
                    print(f"Running {method} | n={nobj}, workers={w}, repeat={r+1}/{repeats}")
                    rows.append(benchmark_one(EXAMPLES_DIR, method, cache[method], w, nobj))

    df = pd.DataFrame(rows)

    # Compute speedup vs. fastest 1-worker runtime per (method, num_objects)
    baseline = (
        df[(df["num_workers"] == 1) & df["runtime_sec"].notna()]
        .groupby(["method", "num_objects"])["runtime_sec"]  # min across repeats
        .min()
    )

    def speedup(row):
        rt = row["runtime_sec"]
        if pd.isna(rt):
            return None
        key = (row["method"], row["num_objects"])
        base = baseline.get(key, None)
        if base is None or base <= 0:
            return None
        return float(base) / float(rt)

    df["speedup"] = df.apply(speedup, axis=1)
    return df


def create_plots(df: pd.DataFrame, outdir: str, prefix: str, repeats: int = 1) -> None:
    """Create grouped bar plots per object count.

    For each distinct value of num_objects in the results DataFrame, generate a
    grouped bar chart where the x-axis lists methods and, for each method, one
    bar per worker count shows the best (minimum) runtime across repeats.

    - One PNG per object count is saved to `outdir` named
      `{prefix}_n{num_objects}.png`.
    - A small textbox on the figure annotates the number of repeats used and the
      logical CPU core count (`os.cpu_count()`).

    Args:
        df: Results DataFrame with columns [method, num_workers, num_objects, runtime_sec, ...].
        outdir: Directory to save figures into (will not be created here; caller ensures it exists).
        prefix: Filename prefix for saved PNGs.
        repeats: Number of repeats per configuration (for annotation only).
    """
    # Ensure we don't modify the original
    data = df.copy()
    if data.empty:
        return

    # Determine distinct object counts
    object_counts = sorted([int(x) for x in data["num_objects"].dropna().unique()])
    if not object_counts:
        return

    # Colors for worker groups (cycled if needed)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for nobj in object_counts:
        sub = data[data["num_objects"] == nobj].copy()
        # Aggregate to best (min) runtime across repeats for plotting
        agg = (
            sub.dropna(subset=["runtime_sec"])  # exclude failed runs from min aggregation
            .groupby(["method", "num_workers"])["runtime_sec"]  # type: ignore[list-item]
            .min()
            .unstack("num_workers")
        )

        # If everything failed for this object count, skip plot
        if agg.empty or agg.notna().sum().sum() == 0:
            continue

        methods = list(agg.index.astype(str))
        # Sorted unique workers present in this subset
        worker_vals = sorted([int(c) for c in agg.columns.tolist() if pd.notna(c)])
        if not worker_vals:
            continue

        x = np.arange(len(methods), dtype=float)
        n_groups = len(worker_vals)
        total_width = 0.8
        bar_width = total_width / max(1, n_groups)

        fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(methods)), 5))

        # Plot each worker group's bars with an offset
        for i, w in enumerate(worker_vals):
            # Retrieve runtimes for this worker; may contain NaN if method failed for this worker
            y = agg.get(w)
            if y is None:
                # Column might be missing entirely; skip
                continue
            y = y.astype(float)
            # Positions centered around x using offset
            offsets = (i - (n_groups - 1) / 2) * bar_width
            pos = x + offsets
            mask = y.notna().to_numpy()
            if mask.any():
                ax.bar(pos[mask], y[mask].to_numpy(), width=bar_width, label=w, color=colors[i % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Runtime in s")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(title="Workers", ncols=min(4, n_groups))

        fig.tight_layout()
        out_path = os.path.join(outdir, f"{prefix}_n{nobj}.png")
        fig.savefig(out_path, dpi=150)
        plt.show(fig)


def parse_args():
    p = argparse.ArgumentParser(description="EnTiSe Examples Benchmark")
    p.add_argument("--methods", type=str, default="all", help="Comma-separated methods or 'all'")
    p.add_argument("--workers", type=str, default="1,4", help="Comma-separated worker counts")
    p.add_argument("--objects", type=str, default="100", help="Comma-separated object counts")
    p.add_argument("--repeats", type=int, default=1, help="Repeats per configuration")
    p.add_argument("--outdir", type=str, default=os.path.join(EXAMPLES_DIR, "benchmarks"))
    p.add_argument("--prefix", type=str, default="benchmark")
    p.add_argument("--plot", type=bool, default=True, help="Save plot PNG (true/false). Default: false")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    discovered = discover_methods(EXAMPLES_DIR)

    if args.methods.strip().lower() == "all":
        methods = discovered
    else:
        requested = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = sorted(set(requested) - set(discovered))
        if unknown:
            print(f"Warning: unknown methods skipped: {', '.join(unknown)}")
        methods = [m for m in requested if m in discovered]

    workers = parse_int_list(args.workers)
    objects_list = parse_int_list(args.objects)

    df = run_sweep(methods, workers, objects_list, repeats=args.repeats)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.outdir, f"{args.prefix}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Optional plotting hook (off by default)
    if args.plot:
        try:
            create_plots(df, args.outdir, args.prefix, repeats=args.repeats)
        except Exception as e:  # noqa: BLE001
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
