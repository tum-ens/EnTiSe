import cProfile
import pstats
import pandas as pd
import numpy as np
from io import StringIO


# Generate synthetic data for benchmarking
def generate_synthetic_data(num_buildings, num_timesteps):
    """
    Generate synthetic building and weather data for benchmarking.
    """
    buildings_df = pd.DataFrame({
        'R': np.random.uniform(0.1, 0.2, num_buildings),
        'C': np.random.uniform(40000, 80000, num_buildings),
        'T_init': np.random.uniform(19, 23, num_buildings),
        'T_min': np.random.uniform(17, 20, num_buildings),
        'T_max': np.random.uniform(24, 27, num_buildings),
        'Heating_Power': np.random.uniform(1500, 3000, num_buildings),
        'Cooling_Power': np.random.uniform(1500, 3000, num_buildings),
        'Solar_Gain': np.random.uniform(0, 100, num_buildings)
    })

    weather_data = pd.DataFrame({
        'Time': pd.date_range(start="2024-01-01", periods=num_timesteps, freq="h"),
        'T_out': np.random.uniform(-5, 10, num_timesteps)
    })

    return buildings_df, weather_data


# Benchmark the simulation
def benchmark_simulation(num_buildings, num_timesteps, output_file="benchmark_stats.txt"):
    """
    Profile the simulation and output benchmark results.
    """
    buildings_df, weather_data = generate_synthetic_data(num_buildings, num_timesteps)

    profiler = cProfile.Profile()
    profiler.enable()

    # Run the simulation
    run_simulation(buildings_df, weather_data)

    profiler.disable()

    # Save benchmark stats to a file
    with open(output_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative').print_stats(20)  # Top 20 functions by cumulative time

    print(f"Benchmark results saved to {output_file}")


# Run the benchmark with different configurations
if __name__ == "__main__":
    timesteps = 8760
    v = 5
    from rc_model5 import run_simulation
    print("Running benchmark with 100 buildings and 24 timesteps...")
    benchmark_simulation(100, timesteps, output_file=f"benchmark_100_buildings_{timesteps}_timesteps_v{v}.txt")

    print("Running benchmark with 1,000 buildings and 24 timesteps...")
    benchmark_simulation(1000, timesteps, output_file=f"benchmark_1000_buildings_{timesteps}_timesteps_v{v}.txt")

    print("Running benchmark with 10,000 buildings and 24 timesteps...")
    benchmark_simulation(10000, timesteps, output_file=f"benchmark_10000_buildings_{timesteps}_timesteps_v{v}.txt")
