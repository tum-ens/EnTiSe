"""
Benchmark script for Heat Pump: Ruhnau

This script tests the performance of the Ruhnau method with 100 heat pump objects.
It generates 100 heat pump objects with varying parameters, uses the Ruhnau method
to generate time series for all objects, and measures the execution time.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from entise.constants import Types
from entise.core.generator import TimeSeriesGenerator


def generate_heat_pump_objects(num_objects=100):
    """
    Generate heat pump objects with varying parameters.

    Args:
        num_objects (int): Number of heat pump objects to generate

    Returns:
        pd.DataFrame: DataFrame containing heat pump objects
    """
    # Create a list to store the objects
    objects_list = []

    # Define heat pump source types
    source_types = ["ASHP", "GSHP", "WSHP"]

    # Define heat sink types
    sink_types = ["floor", "radiator", "water"]

    # Define temperature settings for each sink type
    sink_temps = {"floor": [25, 30, 35], "radiator": [30, 40, 50, 60], "water": [45, 50, 55, 60]}

    # Define gradient settings for each sink type
    sink_gradients = {"floor": -0.5, "radiator": -1.0, "water": 0}

    # Generate objects with varying parameters
    for i in range(num_objects):
        # Generate random variations for parameters
        hp_source = np.random.choice(source_types)
        hp_sink = np.random.choice(sink_types)
        temp_sink = np.random.choice(sink_temps[hp_sink])
        gradient_sink = sink_gradients[hp_sink]
        temp_water = np.random.choice([45, 50, 55, 60])  # DHW temperature
        correction_factor = np.random.uniform(0.8, 0.95)  # Random correction factor between 0.8 and 0.95

        # Create the object
        obj = {
            "id": f"hp_{i+1}",
            "hp": "ruhnau",
            "weather": "weather",
            "hp_source": hp_source,
            "hp_sink": hp_sink,
            "temp_sink": temp_sink,
            "gradient_sink": gradient_sink,
            "temp_water": temp_water,
            "correction_factor": correction_factor,
        }

        # Add the object to the list
        objects_list.append(obj)

    # Convert the list to a DataFrame
    objects_df = pd.DataFrame(objects_list)

    return objects_df


def run_benchmark(num_objects=100, workers=1, visualize=False):
    """
    Run the benchmark with the specified number of objects and workers.

    Args:
        num_objects (int): Number of heat pump objects to generate
        workers (int): Number of workers to use for parallel processing
        visualize (bool): Whether to visualize the results

    Returns:
        tuple: A tuple containing:
            - execution_time (float): Execution time in seconds
            - summary (pd.DataFrame): Summary statistics
            - timeseries (dict): Time series data
    """
    # Generate heat pump objects
    print(f"Generating {num_objects} heat pump objects...")
    objects = generate_heat_pump_objects(num_objects)

    # Load data
    print("Loading data...")
    cwd = "."  # Current working directory
    data = {}
    data_folder = "data"
    for file in os.listdir(os.path.join(cwd, data_folder)):
        if file.endswith(".csv"):
            name = file.split(".")[0]
            data[name] = pd.read_csv(os.path.join(os.path.join(cwd, data_folder, file)), parse_dates=True)
    print("Loaded data keys:", list(data.keys()))

    # Instantiate and configure the generator
    print("Configuring generator...")
    gen = TimeSeriesGenerator()
    gen.add_objects(objects)

    # Generate time series and measure execution time
    print(f"Generating time series with {workers} worker(s)...")
    start_time = time.time()
    summary, df = gen.generate(data, workers=workers)
    end_time = time.time()
    execution_time = end_time - start_time

    # Print execution time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Print summary
    print("Summary:")
    print(summary.head())  # Print only the first few rows

    # Visualize results if requested
    if visualize:
        # Convert index to datetime for all time series
        for obj_id in df:
            if Types.HP in df[obj_id]:
                df[obj_id][Types.HP].index = pd.to_datetime(df[obj_id][Types.HP].index)

        # Get heat pump parameters from objects dataframe
        system_configs = {}
        for _, row in objects.iterrows():
            obj_id = row["id"]
            if obj_id in df:
                hp_source = row["hp_source"] if not pd.isna(row.get("hp_source", pd.NA)) else "Default"
                hp_sink = row["hp_sink"] if not pd.isna(row.get("hp_sink", pd.NA)) else "Default"
                temp_sink = row["temp_sink"] if not pd.isna(row.get("temp_sink", pd.NA)) else "Default"
                temp_water = row["temp_water"] if not pd.isna(row.get("temp_water", pd.NA)) else "Default"
                system_configs[obj_id] = {
                    "hp_source": hp_source,
                    "hp_sink": hp_sink,
                    "temp_sink": temp_sink,
                    "temp_water": temp_water,
                }

        # Figure 1: Histogram of average heating COP
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        heating_cops = []
        for obj_id in df:
            if Types.HP in df[obj_id]:
                heating_col = f"{Types.HP}_{Types.HEATING}"
                if heating_col in df[obj_id][Types.HP].columns:
                    heating_cops.append(df[obj_id][Types.HP][heating_col].mean())

        plt.hist(heating_cops, bins=20)
        plt.title("Histogram of Average Heating COP")
        plt.xlabel("Average COP")
        plt.ylabel("Count")
        plt.grid(axis="y")

        # Figure 2: Histogram of average DHW COP
        plt.subplot(1, 2, 2)
        dhw_cops = []
        for obj_id in df:
            if Types.HP in df[obj_id]:
                dhw_col = f"{Types.HP}_{Types.DHW}"
                if dhw_col in df[obj_id][Types.HP].columns:
                    dhw_cops.append(df[obj_id][Types.HP][dhw_col].mean())

        plt.hist(dhw_cops, bins=20)
        plt.title("Histogram of Average DHW COP")
        plt.xlabel("Average COP")
        plt.ylabel("Count")
        plt.grid(axis="y")

        plt.tight_layout()
        plt.show()

        # Figure 3: Sample of daily profiles for a few systems
        plt.figure(figsize=(12, 8))
        sample_ids = list(df.keys())[:5]  # Take first 5 systems

        for i, obj_id in enumerate(sample_ids):
            if Types.HP not in df[obj_id]:
                continue

            # Get a sample day for heating COP
            heating_col = f"{Types.HP}_{Types.HEATING}"
            if heating_col in df[obj_id][Types.HP].columns:
                sample_day = df[obj_id][Types.HP][heating_col].iloc[:24].copy()

                # Get system parameters for the title
                config = system_configs.get(obj_id, {})
                hp_source = config.get("hp_source", "Default")
                hp_sink = config.get("hp_sink", "Default")

                # Plot the daily profile
                plt.subplot(len(sample_ids), 1, i + 1)
                plt.plot(range(24), sample_day.values, label="Heating COP")

                # Add DHW COP if available
                dhw_col = f"{Types.HP}_{Types.DHW}"
                if dhw_col in df[obj_id][Types.HP].columns:
                    sample_day_dhw = df[obj_id][Types.HP][dhw_col].iloc[:24].copy()
                    plt.plot(range(24), sample_day_dhw.values, label="DHW COP", linestyle="--")

                plt.title(f"ID {obj_id}, Source: {hp_source}, Sink: {hp_sink}")
                plt.xlabel("Hour of Day")
                plt.ylabel("COP")
                plt.legend()
                plt.grid(True)
                plt.xticks(range(0, 24, 2))

        plt.tight_layout()
        plt.show()

    return execution_time, summary, df


if __name__ == "__main__":
    # Run the benchmark with different numbers of workers
    print("Running benchmark with 1 worker...")
    time_1, _, _ = run_benchmark(num_objects=100, workers=1, visualize=False)
    print(f"Runtime: {time_1:.2f} seconds")

    print("\nRunning benchmark with 4 workers...")
    time_4, _, _ = run_benchmark(num_objects=100, workers=4, visualize=False)
    print(f"Runtime: {time_4:.2f} seconds")

    # Print speedup
    speedup = time_1 / time_4
    print(f"\nSpeedup with 4 workers: {speedup:.2f}x")

    # Run with visualization for the last run
    print("\nRunning benchmark with visualization...")
    _, _, _ = run_benchmark(num_objects=10, workers=4, visualize=True)
