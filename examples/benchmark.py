import time
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import platform
from matplotlib.ticker import MaxNLocator

import os
import sys

# Repo root (…/entise)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Absolute examples directory (…/entise/examples)
EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))


# --------------------------------------------------
# Run benchmark for ONE method
# --------------------------------------------------
def benchmark_method(method_name: str, num_workers: int, num_objects: int):

    method_path = f"examples.{method_name}.runme"
    module = importlib.import_module(method_path)

    get_input = module.get_input
    simulate = module.simulate

    # IMPORTANT: use absolute path so running from any cwd works
    base_path = os.path.join(EXAMPLES_DIR, method_name)

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
def create_barplot(df: pd.DataFrame, num_objects: int, out_png="benchmark_results.png"):
    import platform
    from matplotlib.ticker import MaxNLocator

    plot_df = (
        df.pivot_table(index="method", columns="num_workers", values="runtime_sec", aggfunc="mean")
        .sort_index()
    )

    methods = plot_df.index.tolist()
    workers = plot_df.columns.tolist()

    # -------- broken axis settings --------
    y_break = 12.0          # seconds – adjust if needed
    upper_margin = 1.05

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(16, 7),
        gridspec_kw={"height_ratios": [1, 3]}
    )

    x = range(len(methods))
    total_width = 0.78
    bar_width = total_width / len(workers)
    offsets = [(-total_width / 2) + (i + 0.5) * bar_width for i in range(len(workers))]

    ymax = plot_df.max().max()

    for i, w in enumerate(workers):
        heights = plot_df[w].values
        xs = [xi + offsets[i] for xi in x]

        ax_bot.bar(xs, heights, width=bar_width, label=str(w))
        ax_top.bar(xs, heights, width=bar_width)

        for xb, runtime in zip(xs, heights):
            if runtime <= y_break:
                itps = num_objects / runtime
                ax_bot.text(xb, runtime * 1.02, f"{itps:.1f}it/s",
                            ha="center", va="bottom", fontsize=8)

    # ---- axis limits ----
    ax_bot.set_ylim(0, y_break)
    ax_top.set_ylim(y_break * upper_margin, ymax * 1.05)

    # ---- formatting ----
    ax_bot.set_ylabel("Simulation time [s]")
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(methods, fontsize=9)
    ax_bot.grid(axis="y", alpha=0.3)
    ax_bot.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax_top.grid(axis="y", alpha=0.3)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)

    # ---- diagonal break marks ----
    d = 0.01
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # ---- legend + info ----
    ax_bot.legend(loc="best", frameon=False)

    uname = platform.uname()
    cpu = platform.processor() or uname.processor or uname.machine
    info = f"n: {num_objects}\n{uname.system} {uname.release}\nCPU: {cpu}"

    ax_top.text(
        0.98, 0.95, info,
        transform=ax_top.transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", edgecolor="black"),
        fontsize=9
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

    create_barplot(df, num_objects)

    print("\n✅ Benchmark completed")
    print(df)