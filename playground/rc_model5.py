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

    return indoor_temperature, heating_load, cooling_load


def simulate_building(params, outdoor_temperature, timestep):
    """
    Simulates a single building.
    """
    indoor_temperature, heating_load, cooling_load = calculate_1R1C_optimized(
        params['R'],
        params['C'],
        params['T_init'],
        outdoor_temperature,
        params.get('Heating_Power', np.inf),
        params.get('Cooling_Power', np.inf),
        params.get('Solar_Gain', 0),
        params['T_min'],
        params['T_max'],
        timestep,
    )

    return pd.DataFrame({
        'Time': params['time_index'],
        'Indoor_Temperature': indoor_temperature,
        'Heating_Load': heating_load,
        'Cooling_Load': cooling_load,
    })


def process_buildings_parallel(buildings_df, weather_data, timestep=3600):
    """
    Processes multiple buildings in parallel using joblib with tqdm integration.
    """
    # Precompute outdoor temperature array
    outdoor_temperature = weather_data['T_out'].to_numpy(dtype=np.float32)

    # Add a time index to each building's parameters
    building_params = buildings_df.to_dict('records')
    for building in building_params:
        building['time_index'] = weather_data['Time']

    # Use Parallel with a tqdm progress bar
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(simulate_building)(params, outdoor_temperature, timestep)
        for params in tqdm(building_params, desc="Simulating Buildings", total=len(building_params))
    )

    return results


def run_simulation(buildings_df, weather_data, timestep=3600):
    """
    Main simulation entry point.
    """
    return process_buildings_parallel(buildings_df, weather_data, timestep)


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
