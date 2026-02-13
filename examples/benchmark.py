"""
Benchmark runner for EnTiSe example methods.

This module can be used either via a CLI (subcommands) or programmatically
(via `run_benchmark()` or `main(argv=...)`).

Subcommands:
- run:  Discover example method folders under examples/ (any subfolder containing an objects.csv),
        run benchmarks, save a timestamped CSV, and optionally generate plots.
- plot: Generate plots from an existing benchmark results CSV (no rerun).
- list: List discovered example methods (optionally filtered by a glob).

Key features:
- Methods can be selected using:
  - 'all' (default, all discovered example methods)
  - exact names (e.g., 'hp_ruhnau')
  - glob patterns (e.g., 'hp_*', '*_pvlib')
  - comma-separated lists are accepted in addition to space-separated tokens
- Workers and objects accept space-separated lists (preferred) and also comma-separated lists.
- Each benchmark configuration is repeated `--repeats` times and recorded with an explicit `repeat` column.
- Output CSV columns:
  method, num_workers, num_objects, repeat, runtime_sec, iter_per_sec, speedup, status, error
- Speedup is computed relative to the mean 1-worker runtime for each (method, num_objects).
- Plots:
  - One figure per `num_objects`
  - Bar heights show mean runtime across repeats
  - If repeats > 1, per-repeat runtimes are overlaid as scatter points
  - A summary table below the chart reports mean runtime, mean iter/s, and mean speedup per worker
  - Figures are saved as PDF to `{outdir}/{prefix}_n{num_objects}.pdf`

Programmatic usage:
- Run benchmarks and get a DataFrame + CSV path:
    df, csv_path = run_benchmark(
        methods=["all"], workers=[1, 4], objects=[100], repeats=3,
        outdir="examples/benchmarks", prefix="benchmark", plot=True, show=False
    )
- Call the CLI programmatically:
    main(["run", "--methods", "hp_*", "--workers", "1", "4", "--objects", "100", "--repeats", "3", "--plot"])

Usage examples (from repository root):

List methods:
- python examples/benchmark.py list
- python examples/benchmark.py list --filter "hp_*"

Run a benchmark grid:
- python examples/benchmark.py run --methods all --workers 1 2 4 --objects 50 100 --repeats 3

Run and plot:
- python examples/benchmark.py run --methods all --workers 1 2 4 --objects 50 100 --repeats 3 --plot

Run only a subset using globs:
- python examples/benchmark.py run --methods "pv_*" "hp_*" --workers 1 4 --objects 100 --repeats 5 --plot

Plot from an existing CSV:
- python examples/benchmark.py plot --results examples/benchmarks/benchmark_YYYYMMDD-HHMMSS.csv

Plot with custom output directory/prefix:
- python examples/benchmark.py plot --results examples/benchmarks/benchmark_YYYYMMDD-HHMMSS.csv --outdir examples/benchmarks --prefix myplots

Backwards-friendly comma-separated inputs:
- python examples/benchmark.py run --methods pv_pvlib,hp_ruhnau --workers 1,2,4 --objects 50,100 --repeats 3 --plot
"""


import argparse
import fnmatch
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root is importable
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from examples.utils import load_input, run_simulation  # noqa: E402

EXAMPLES_DIR = THIS_DIR


def _split_csv_or_space(values: Optional[Sequence[str]]) -> List[str]:
    """
    Accept either:
    - multiple tokens via nargs="+", e.g. ["1", "2", "4"]
    - a single comma-separated token, e.g. ["1,2,4"]
    - a mixture, e.g. ["1,2", "4"]
    """
    if not values:
        return []
    out: List[str] = []
    for v in values:
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        out.extend(parts if parts else [str(v).strip()])
    return [x for x in out if x]


def parse_int_list(values: Optional[Sequence[str]]) -> List[int]:
    toks = _split_csv_or_space(values)
    return [int(x) for x in toks]


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


def select_methods(discovered: List[str], selectors: Sequence[str]) -> List[str]:
    """
    selectors can include:
    - 'all'
    - exact method names
    - glob patterns like 'hp_*' or '*_pvlib'
    - comma-separated lists (already split by caller helpers)

    Unknown tokens are ignored with a warning at the caller.
    """
    if not selectors:
        return discovered

    sel = [s.strip() for s in selectors if s.strip()]
    if any(s.lower() == "all" for s in sel):
        return discovered

    matched: List[str] = []
    for token in sel:
        if any(ch in token for ch in ["*", "?", "["]):
            matched.extend([m for m in discovered if fnmatch.fnmatch(m, token)])
        else:
            if token in discovered:
                matched.append(token)

    # De-dup preserving order
    seen = set()
    out: List[str] = []
    for m in matched:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


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
    method: str,
    cached: Tuple[pd.DataFrame, Dict[str, object]],
    workers: int,
    num_objects: int,
    repeat: int,
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
            "repeat": repeat,
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
            "repeat": repeat,
            "runtime_sec": None,
            "iter_per_sec": None,
            "status": "failed",
            "error": str(e),
        }


def run_sweep(methods: List[str], workers: List[int], objects_list: List[int], repeats: int) -> pd.DataFrame:
    cache: Dict[str, Tuple[pd.DataFrame, Dict[str, object]]] = {}
    rows: List[Dict[str, object]] = []

    for method in methods:
        base_path = os.path.join(EXAMPLES_DIR, method)
        try:
            cache[method] = load_input(base_path, load_common_data=True)
        except Exception as e:  # noqa: BLE001
            for nobj in objects_list:
                for w in workers:
                    for r in range(1, repeats + 1):
                        rows.append(
                            {
                                "method": method,
                                "num_workers": w,
                                "num_objects": nobj,
                                "repeat": r,
                                "runtime_sec": None,
                                "iter_per_sec": None,
                                "status": "failed",
                                "error": f"load_input: {e}",
                            }
                        )
            continue

        for nobj in objects_list:
            for w in workers:
                for r in range(1, repeats + 1):
                    print(f"Running {method} | n={nobj}, workers={w}, repeat={r}/{repeats}")
                    rows.append(benchmark_one(method, cache[method], w, nobj, repeat=r))

    df = pd.DataFrame(rows)

    baseline = (
        df[(df["num_workers"] == 1) & df["runtime_sec"].notna()]
        .groupby(["method", "num_objects"])["runtime_sec"]
        .mean()
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


def _format_method_label(method: str) -> str:
    if "_" not in method:
        return method
    ts_type, method_name = method.split("_", 1)
    return f"{ts_type}\n{method_name}"


def create_plots(df: pd.DataFrame, outdir: str, prefix: str, repeats: Optional[int] = None, show: bool = True) -> None:
    data = df.copy()
    if data.empty:
        return

    object_counts = sorted([int(x) for x in data["num_objects"].dropna().unique()])
    if not object_counts:
        return

    # Okabe–Ito (colorblind-safe, "designed" look)
    # https://jfly.uni-koeln.de/color/
    palette = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#CC79A7",  # purple
        "#56B4E9",  # sky blue
        "#000000",  # black
        "#F0E442",  # yellow
    ]

    if repeats is None and "repeat" in data.columns:
        rep_max = pd.to_numeric(data["repeat"], errors="coerce").max()
        repeats = int(rep_max) if pd.notna(rep_max) else None

    def _fmt(x: float, nd: int = 2) -> str:
        if pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}"

    for nobj in object_counts:
        sub = data[(data["num_objects"] == nobj) & data["runtime_sec"].notna()].copy()
        if sub.empty:
            continue

        mean_rt = (
            sub.groupby(["method", "num_workers"], as_index=False)["runtime_sec"]
            .mean()
            .pivot(index="method", columns="num_workers", values="runtime_sec")
        )
        if mean_rt.empty or mean_rt.notna().sum().sum() == 0:
            continue

        mean_iter = (
            sub.groupby(["method", "num_workers"], as_index=False)["iter_per_sec"]
            .mean()
            .pivot(index="method", columns="num_workers", values="iter_per_sec")
        )

        mean_sp = (
            sub.groupby(["method", "num_workers"], as_index=False)["speedup"]
            .mean()
            .pivot(index="method", columns="num_workers", values="speedup")
        )

        methods = list(mean_rt.index.astype(str))
        worker_vals = sorted([int(c) for c in mean_rt.columns.tolist() if pd.notna(c)])
        if not worker_vals:
            continue

        x = np.arange(len(methods), dtype=float)
        n_groups = len(worker_vals)
        total_width = 0.8
        bar_width = total_width / max(1, n_groups)

        fig_w = max(10, 1.35 * len(methods))
        fig_h = 6.4
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3.6, 1.8], hspace=0.06)

        ax = fig.add_subplot(gs[0])
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis("off")

        # Subtle axis styling
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_alpha(0.35)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)

        method_to_idx = {m: j for j, m in enumerate(methods)}

        # Bars
        for i, w in enumerate(worker_vals):
            y = mean_rt.get(w)
            if y is None:
                continue
            y = y.astype(float)

            offsets = (i - (n_groups - 1) / 2) * bar_width
            pos = x + offsets
            mask = y.notna().to_numpy()

            if mask.any():
                color = palette[i % len(palette)]
                ax.bar(
                    pos[mask],
                    y[mask].to_numpy(),
                    width=bar_width,
                    label=str(w),
                    color=color,
                    alpha=0.92,
                    edgecolor=mcolors.to_rgba("black", 0.15),
                    linewidth=0.6,
                )

                # Optional: plot repeat points only if repeats > 1
                if repeats and repeats > 1:
                    pts = sub[sub["num_workers"] == w].copy()
                    pts["method"] = pts["method"].astype(str)
                    xs = np.array([method_to_idx[m] for m in pts["method"].tolist()], dtype=float) + offsets
                    ax.scatter(
                        xs,
                        pts["runtime_sec"].to_numpy(dtype=float),
                        s=14,
                        color=mcolors.to_rgba(color, 0.65),
                        linewidths=0.0,
                        zorder=3,
                    )

        ax.set_ylabel("Runtime [s]")
        ax.set_xticks(x)
        ax.set_xticklabels([_format_method_label(m) for m in methods], rotation=0, ha="center")

        # Legend
        leg = ax.legend(title="Workers", loc="upper right", framealpha=0.92, borderpad=0.6)
        leg.get_frame().set_edgecolor(mcolors.to_rgba("black", 0.15))
        leg.get_frame().set_linewidth(0.8)

        # ---------- Table ----------
        rows: list[str] = []
        cell_text: list[list[str]] = []

        for w in worker_vals:
            rows.append(f"Runtime [s] (w={w})")
            cell_text.append(
                [
                    _fmt(mean_rt.loc[m, w] if (m in mean_rt.index and w in mean_rt.columns) else np.nan, 2)
                    for m in methods
                ]
            )

        for w in worker_vals:
            rows.append(f"Iter/s (w={w})")
            cell_text.append(
                [
                    _fmt(mean_iter.loc[m, w] if (m in mean_iter.index and w in mean_iter.columns) else np.nan, 2)
                    for m in methods
                ]
            )

        if len(worker_vals) > 1:
            for w in worker_vals:
                if w == 1:
                    continue
                rows.append(f"Speedup vs w=1 (w={w})")
                cell_text.append(
                    [
                        _fmt(mean_sp.loc[m, w] if (m in mean_sp.index and w in mean_sp.columns) else np.nan, 2)
                        for m in methods
                    ]
                )

        tbl = ax_tbl.table(
            cellText=cell_text,
            rowLabels=rows,
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.15)

        # Table styling (thin borders + zebra rows + subtle row-label emphasis)
        cells = tbl.get_celld()

        for (r, c), cell in cells.items():
            cell.set_linewidth(0.6)
            cell.set_edgecolor(mcolors.to_rgba("black", 0.18))

            # Zebra striping for metric rows (data rows start at r=0)
            if r % 2 == 1:
                cell.set_facecolor(mcolors.to_rgba("#000000", 0.03))
            else:
                cell.set_facecolor("white")

            # Row label column is c == -1
            if c == -1:
                cell.get_text().set_ha("left")
                cell.get_text().set_color(mcolors.to_rgba("black", 0.85))
                cell.set_facecolor(mcolors.to_rgba("#000000", 0.06))  # slightly stronger for row labels

        # Save
        out_path = os.path.join(outdir, f"{prefix}_n{nobj}.pdf")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EnTiSe examples benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List discovered example methods")
    p_list.add_argument("--filter", type=str, default=None, help="Optional glob filter like 'hp_*'")

    # run
    p_run = sub.add_parser("run", help="Run benchmarks and write results CSV (optionally plot)")
    p_run.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Method names or globs (space-separated). Use 'all' or patterns like 'hp_*'. Commas also accepted.",
    )
    p_run.add_argument(
        "--workers", nargs="+", default=["1", "4"], help="Worker counts. Space-separated or comma-separated."
    )
    p_run.add_argument(
        "--objects", nargs="+", default=["100"], help="Object counts. Space-separated or comma-separated."
    )
    p_run.add_argument("--repeats", type=int, default=1, help="Repeats per configuration")
    p_run.add_argument("--outdir", type=str, default=os.path.join(EXAMPLES_DIR, "benchmarks"))
    p_run.add_argument("--prefix", type=str, default="benchmark")
    p_run.add_argument("--plot", action="store_true", help="If set, create and save plots after run")

    # plot
    p_plot = sub.add_parser("plot", help="Plot from an existing benchmark results CSV")
    p_plot.add_argument("--results", type=str, required=True, help="Path to a results CSV from a previous run")
    p_plot.add_argument("--outdir", type=str, default=os.path.join(EXAMPLES_DIR, "benchmarks"))
    p_plot.add_argument("--prefix", type=str, default="benchmark")

    return p


def cmd_list(args) -> int:
    discovered = discover_methods(EXAMPLES_DIR)
    if args.filter:
        discovered = [m for m in discovered if fnmatch.fnmatch(m, args.filter)]
    for m in discovered:
        print(m)
    return 0


def cmd_run(args) -> int:
    os.makedirs(args.outdir, exist_ok=True)

    discovered = discover_methods(EXAMPLES_DIR)

    methods_tokens = _split_csv_or_space(args.methods)
    methods = select_methods(discovered, methods_tokens)

    # Warn about unknown non-glob explicit tokens
    unknown = []
    for token in methods_tokens:
        is_glob = any(ch in token for ch in ["*", "?", "["])
        if (not is_glob) and (token.lower() != "all") and (token not in discovered):
            unknown.append(token)
    if unknown:
        print(f"Warning: unknown methods skipped: {', '.join(sorted(set(unknown)))}")

    workers = parse_int_list(args.workers)
    objects_list = parse_int_list(args.objects)

    if not methods:
        raise SystemExit("No methods selected.")
    if not workers:
        raise SystemExit("No workers specified.")
    if not objects_list:
        raise SystemExit("No objects specified.")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be >= 1")

    df = run_sweep(methods, workers, objects_list, repeats=args.repeats)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(args.outdir, f"{args.prefix}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    if args.plot:
        create_plots(df, args.outdir, args.prefix, repeats=args.repeats)
    return 0


def cmd_plot(args) -> int:
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.results)
    create_plots(df, args.outdir, args.prefix, repeats=None)
    return 0


def run_benchmark(
    methods: list = ("all"),
    workers: list = (1, 4),
    objects: list = (100),
    repeats: int = 1,
    outdir="benchmarks",
    prefix="benchmark",
    plot=False,
    show=False,
):
    os.makedirs(outdir, exist_ok=True)
    df = run_sweep(methods, workers, objects, repeats)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join(outdir, f"{prefix}_{ts}.csv")
    df.to_csv(csv_path, index=False)

    if plot:
        create_plots(df, outdir, prefix, repeats=repeats, show=show)

    return df, csv_path


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "plot":
        return cmd_plot(args)

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
