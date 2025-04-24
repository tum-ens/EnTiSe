import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def calculate_1R1C(building_row, weather_data, timestep, output_vars):
    """
    Calculates the thermal response of a single building using the 1R1C model.
    """
    # Extract building parameters
    thermal_resistance = building_row['R']  # Thermal resistance [K/W]
    thermal_capacitance = building_row['C']  # Thermal capacitance [J/K]
    initial_temperature = building_row['T_init']  # Initial indoor temperature [°C]
    outdoor_temperature = weather_data['T_out']  # Outdoor temperature time series [°C]
    heating_power = building_row.get('Heating_Power', np.inf)  # Max heating power [W]
    cooling_power = building_row.get('Cooling_Power', np.inf)  # Max cooling power [W]
    solar_gain = building_row.get('Solar_Gain', 0)  # Solar gain [W]

    # Time series variables
    n_steps = len(outdoor_temperature)
    indoor_temperature = np.zeros(n_steps)
    heating_load = np.zeros(n_steps)
    cooling_load = np.zeros(n_steps)

    # Initialize indoor temperature
    indoor_temperature[0] = initial_temperature

    # Simulation loop
    for t in range(1, n_steps):
        delta_t = timestep
        temp_change = (
            (outdoor_temperature[t] - indoor_temperature[t - 1]) / thermal_resistance
            + solar_gain
        ) * delta_t / thermal_capacitance

        indoor_temperature[t] = indoor_temperature[t - 1] + temp_change

        # Apply heating and cooling limits
        if indoor_temperature[t] < building_row['T_min']:
            heating_load[t] = min(heating_power, (building_row['T_min'] - indoor_temperature[t]) / delta_t)
            indoor_temperature[t] = building_row['T_min']
        elif indoor_temperature[t] > building_row['T_max']:
            cooling_load[t] = min(cooling_power, (indoor_temperature[t] - building_row['T_max']) / delta_t)
            indoor_temperature[t] = building_row['T_max']

    # Compile results
    result = pd.DataFrame({
        'Time': weather_data['Time'],
        'Indoor_Temperature': indoor_temperature,
        'Heating_Load': heating_load,
        'Cooling_Load': cooling_load
    })

    return result[output_vars]

def process_buildings_parallel(buildings_df, weather_data, timestep, output_vars):
    """
    Processes multiple buildings in parallel.
    """
    n_buildings = len(buildings_df)
    with Pool(cpu_count()) as pool:
        func = partial(calculate_1R1C, weather_data=weather_data, timestep=timestep, output_vars=output_vars)
        results = list(tqdm(pool.imap(func, [row for _, row in buildings_df.iterrows()]), total=n_buildings))
    return results

def run_simulation(buildings_df, weather_data, timestep=3600, output_vars=None):
    """
    Main simulation function.
    """
    if output_vars is None:
        output_vars = ['Time', 'Indoor_Temperature', 'Heating_Load', 'Cooling_Load']

    results = process_buildings_parallel(buildings_df, weather_data, timestep, output_vars)
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
