import time
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import platform
from matplotlib.ticker import MaxNLocator

import os
import sys

# Ensure repo root is on PYTHONPATH so `examples.*` imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------
# Run benchmark for ONE method
# --------------------------------------------------
def benchmark_method(method_name: str, num_workers: int, num_objects: int):

    method_path = f"examples.{method_name}.runme"
    module = importlib.import_module(method_path)

    get_input = module.get_input
    simulate = module.simulate

    base_path = os.path.join("examples", method_name)

    # Load base input
    objects, data = get_input(base_path)

    # Scale objects
    objects = pd.concat(
        [objects] * ((num_objects // len(objects)) + 1),
        ignore_index=True
    ).iloc[:num_objects]

    # Run & time
    start = time.perf_counter()
    simulate(objects, data, workers=num_workers, path=base_path, export=False, plot=False)
    end = time.perf_counter()

    runtime = end - start

    return {
        "method": method_name,
        "num_workers": num_workers,
        "num_objects": num_objects,
        "runtime_sec": runtime,
        "runtime_per_object": runtime / num_objects,
    }


# --------------------------------------------------
# Run full benchmark sweep
# --------------------------------------------------
def run_benchmarks(methods, num_workers_list, num_objects):
    results = []

    for method in methods:
        for workers in num_workers_list:
            print(f"Running {method} | objects={num_objects}, workers={workers}")
            res = benchmark_method(method, workers, num_objects)
            results.append(res)

    return pd.DataFrame(results)


# --------------------------------------------------
# Output helpers
# --------------------------------------------------
def create_report(df):
    df.to_csv("benchmark_results.csv", index=False)


def create_barplot(df: pd.DataFrame, num_objects: int, out_png="benchmark_results.png"):
    """
    Grouped bar chart:
    - x: method
    - grouped bars: num_workers
    - y: runtime_sec
    - annotation: it/s = num_objects / runtime_sec
    """

    plot_df = (
        df.pivot_table(index="method", columns="num_workers", values="runtime_sec", aggfunc="mean")
        .sort_index()
    )

    methods = plot_df.index.tolist()
    workers = plot_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(16, 6))

    x = range(len(methods))
    total_width = 0.78
    bar_width = total_width / len(workers)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(len(workers))]

    ymax = plot_df.max().max()

    for i, w in enumerate(workers):
        heights = plot_df[w].values
        bars = ax.bar(
            [xi + offsets[i] for xi in x],
            heights,
            width=bar_width,
            label=str(w),
        )

        for b, runtime in zip(bars, heights):
            itps = num_objects / runtime
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01 * ymax,
                f"{itps:.1f}it/s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Simulation time [s]")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    cpu = platform.processor() or "Unknown CPU"
    info = f"n: {num_objects}\nCPU: {cpu}"
    ax.text(
        0.98,
        0.95,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black"),
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"📊 Saved benchmark plot: {out_png}")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    methods = [
        "pv_pvlib",
        "hp_ruhnau",
        "dhw_jordanvajen",
        "wind_wplib",
        "hvac_1r1c",
        "hvac_5r1c",
        "hvac_7r2c",
    ]

    num_workers = [1, 2, 4]
    num_objects = 100

    df = run_benchmarks(methods, num_workers, num_objects)

    create_report(df)
    create_barplot(df, num_objects)

    print("\n✅ Benchmark completed")
    print(df)