import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit


@njit
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
    n_steps = len(outdoor_temperature)
    indoor_temperature = np.zeros(n_steps, dtype=np.float32)
    heating_load = np.zeros(n_steps, dtype=np.float32)
    cooling_load = np.zeros(n_steps, dtype=np.float32)

    # Initialize indoor temperature
    indoor_temperature[0] = initial_temperature

    for t in range(1, n_steps):
        # Temperature change from thermal dynamics
        temp_change = (
            (outdoor_temperature[t] - indoor_temperature[t - 1]) / thermal_resistance
            + solar_gain
        ) * timestep / thermal_capacitance

        indoor_temperature[t] = indoor_temperature[t - 1] + temp_change

        # Apply heating or cooling if needed
        if indoor_temperature[t] < T_min:
            heating_load[t] = min(heating_power, (T_min - indoor_temperature[t]) / timestep)
            indoor_temperature[t] = T_min
        elif indoor_temperature[t] > T_max:
            cooling_load[t] = min(cooling_power, (indoor_temperature[t] - T_max) / timestep)
            indoor_temperature[t] = T_max

    return indoor_temperature, heating_load, cooling_load


def process_single_building(building_row, weather_data, timestep):
    """
    Processes a single building by applying the 1R1C model.
    """
    # Extract parameters from the building data row
    thermal_resistance = building_row['R']
    thermal_capacitance = building_row['C']
    initial_temperature = building_row['T_init']
    heating_power = building_row.get('Heating_Power', np.inf)
    cooling_power = building_row.get('Cooling_Power', np.inf)
    solar_gain = building_row.get('Solar_Gain', 0)
    T_min = building_row['T_min']
    T_max = building_row['T_max']

    # Convert weather data to NumPy for performance
    outdoor_temperature = weather_data['T_out'].to_numpy(dtype=np.float32)

    # Run the optimized simulation
    indoor_temperature, heating_load, cooling_load = calculate_1R1C_optimized(
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
    )

    # Return results as a DataFrame
    return pd.DataFrame({
        'Time': weather_data['Time'],
        'Indoor_Temperature': indoor_temperature,
        'Heating_Load': heating_load,
        'Cooling_Load': cooling_load,
    })


def process_buildings_parallel(buildings_df, weather_data, timestep=3600):
    """
    Processes multiple buildings in parallel using joblib.
    """
    results = Parallel(n_jobs=-1)(
        delayed(process_single_building)(row, weather_data, timestep)
        for _, row in tqdm(buildings_df.iterrows(), total=len(buildings_df))
    )
    return results


def run_simulation(buildings_df, weather_data, timestep=3600):
    """
    Main simulation entry point.
    """
    results = process_buildings_parallel(buildings_df, weather_data, timestep)
    return results


def main() -> None:
    # Example input data
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

    periods = 8760
    weather_data = pd.DataFrame({
        'Time': pd.date_range(start="2024-01-01", periods=periods, freq="H"),
        'T_out': [-5 + i * 0.5 for i in range(periods)]
    })

    # Run simulation
    results = run_simulation(buildings_df, weather_data)

    # Output
    for idx, result in enumerate(results):
        print(f"Building {idx + 1} Results:")
        print(result)


if __name__ == '__main__':
    main()
