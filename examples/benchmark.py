
import time
import importlib
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

# Ensure repo root is on PYTHONPATH so `examples.*` imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# --------------------------------------------------
# Run benchmark for ONE method
# --------------------------------------------------
def benchmark_method(method_name: str, num_workers: int, num_objects: int):
    """
    Runs benchmark for a single method.
    """

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

    print("\n✅ Benchmark completed")
    print(df)