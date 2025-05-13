import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def calculate_1R1C_optimized(
        thermal_resistance,
        thermal_capacitance,
        initial_temperature,
        outdoor_temperature,
        heating_power,
        cooling_power,
        solar_gain,
        T_min,
        T_max,
        timestep,
):
    """
    Optimized calculation of the 1R1C thermal model for a single building.
    """
    # print(f"Number of timesteps: {len(outdoor_temperature)}")
    n_steps = len(outdoor_temperature)
    indoor_temperature = np.zeros(n_steps, dtype=np.float32)
    heating_load = np.zeros(n_steps, dtype=np.float32)
    cooling_load = np.zeros(n_steps, dtype=np.float32)

    # Initialize indoor temperature
    indoor_temperature[0] = initial_temperature

    for t in range(1, n_steps):
        temp_change = (
                              (outdoor_temperature[t] - indoor_temperature[t - 1]) / thermal_resistance
                              + solar_gain
                      ) * timestep / thermal_capacitance

        indoor_temperature[t] = indoor_temperature[t - 1] + temp_change

        if indoor_temperature[t] < T_min:
            heating_load[t] = min(heating_power, (T_min - indoor_temperature[t]) / timestep)
            indoor_temperature[t] = T_min
        elif indoor_temperature[t] > T_max:
            cooling_load[t] = min(cooling_power, (indoor_temperature[t] - T_max) / timestep)
            indoor_temperature[t] = T_max

    # print("Finished 1R1C calculation.")
    return indoor_temperature, heating_load, cooling_load


def process_building(index, building_data, weather_data):
    """
    Process a single building without shared memory.
    """
    # print(f"Processing building {index}...")
    params = building_data

    indoor_temperature, heating_load, cooling_load = calculate_1R1C_optimized(
        params['R'],
        params['C'],
        params['T_init'],
        weather_data['T_out'].to_numpy(dtype=np.float32),
        params.get('Heating_Power', np.inf),
        params.get('Cooling_Power', np.inf),
        params.get('Solar_Gain', 0),
        params['T_min'],
        params['T_max'],
        params['timestep']
    )

    # print(f"Finished processing building {index}.")
    return pd.DataFrame({
        'Time': weather_data['Time'],
        'Indoor_Temperature': indoor_temperature,
        'Heating_Load': heating_load,
        'Cooling_Load': cooling_load,
    })


def process_buildings_parallel(buildings_df, weather_data, timestep=3600, batch_size=10):
    """
    Process multiple buildings in parallel with batching.
    """
    # print(f"Starting parallel processing for {len(buildings_df)} buildings...")

    # Prepare building data
    building_data = [
        {**row, 'timestep': timestep} for _, row in buildings_df.iterrows()
    ]

    # Group buildings into batches
    batched_data = [
        building_data[i:i + batch_size] for i in range(0, len(building_data), batch_size)
    ]
    # print(f"Number of batches: {len(batched_data)}")

    results = []
    try:
        with Pool(cpu_count()) as pool:
            for batch in tqdm(batched_data):
                batch_results = pool.starmap(
                    process_building,
                    [(i, data, weather_data) for i, data in enumerate(batch)]
                )
                results.extend(batch_results)
    except Exception as e:
        # print(f"Error during parallel processing: {e}")
        raise

    # print("All buildings processed successfully.")
    return results


def run_simulation(buildings_df, weather_data, timestep=3600):
    """
    Main simulation entry point.
    """
    # print("Starting simulation...")
    results = process_buildings_parallel(buildings_df, weather_data, timestep)
    # print("Simulation completed.")
    return results


# Example Usage
if __name__ == "__main__":
    # Example building data
    buildings_df = pd.DataFrame({
        'R': [0.1, 0.15],
        'C': [50000, 70000],
        'T_init': [20, 22],
        'T_min': [18, 20],
        'T_max': [24, 26],
        'Heating_Power': [2000, 2500],
        'Cooling_Power': [2000, 2500],
        'Solar_Gain': [0, 50]
    })

    # Example weather data
    weather_data = pd.DataFrame({
        'Time': pd.date_range(start="2024-01-01", periods=24, freq="H"),
        'T_out': [-5 + i * 0.5 for i in range(24)]
    })

    # Run the simulation
    results = run_simulation(buildings_df, weather_data)

    # Print results for each building
    for idx, result in enumerate(results):
        print(f"Building {idx + 1} Results:")
        print(result)
