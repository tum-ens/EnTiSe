#!/usr/bin/env python
import numpy as np
import pandas as pd
import datetime as dt
import time
import os

# -------------------------------
# Settings and constants
# -------------------------------
SIM_START_DATE = '2020-01-01'
SIM_END_DATE = '2020-12-31'  # End date non-inclusive
RESOLUTION_MIN = 15  # Time resolution in minutes

DEFAULT_TW = 60  # Outlet temperature (°C)
DEFAULT_TCOLD = 10  # Inlet temperature (°C)
DENSITY = 1000  # Water density [kg/m³]
CP = 4186  # Specific heat capacity [J/(kg·°C)]

# Seasonal factor parameters
SEASONAL_AMPLITUDE = 0.1  # ±10% variation
SEASONAL_PHASE = 1  # Day-of-year when factor is maximum (e.g., 1 = January 1st)

CSV_FILE = 'dhw_activity_adjusted.csv'
RANDOM_SEED = 42


# -------------------------------
# Utility functions
# -------------------------------
def parse_time_str(time_str: str) -> dt.time:
    """Parse a HH:MM:SS string to a time object."""
    return dt.datetime.strptime(time_str, '%H:%M:%S').time()


def convert_time_to_seconds(t: dt.time) -> int:
    """Convert a time object to seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second


def seasonal_vector(doy_array: np.ndarray, amplitude: float, phase: float) -> np.ndarray:
    """
    Vectorized seasonal multiplier: returns an array of multipliers for given day-of-year values.
    Uses the formula: 1 + amplitude * cos(2 * pi * (doy - phase) / 365)
    """
    return 1 + amplitude * np.cos(2 * np.pi * (doy_array - phase) / 365)


# -------------------------------
# Main simulation function (vectorized approach)
# -------------------------------
def simulate_dhw_series_vectorized(activity_csv: str,
                                   sim_start: str,
                                   sim_end: str,
                                   res_min: int,
                                   Tw: float = DEFAULT_TW,
                                   Tcold: float = DEFAULT_TCOLD,
                                   seed: int = None,
                                   seasonal_amplitude: float = SEASONAL_AMPLITUDE,
                                   seasonal_phase: float = SEASONAL_PHASE) -> (pd.Series, pd.Series):
    """
    Generate DHW consumption (m³) and energy (Wh) timeseries
    using probabilistic events (with seasonal scaling) defined in a CSV.

    The CSV must contain columns:
       day, time, event, probability, duration, sigma_duration, flow_rate, sigma_flow_rate
    Duration and sigma_duration are in seconds.
    Flow_rate and sigma_flow_rate are in liters per second.
    """
    if seed is not None:
        np.random.seed(seed)

    # Load events from CSV and convert times to seconds since midnight
    if not os.path.exists(activity_csv):
        raise FileNotFoundError(f"CSV file {activity_csv} not found.")
    df_events = pd.read_csv(activity_csv)
    df_events['time_obj'] = df_events['time'].apply(parse_time_str)
    df_events['time_sec'] = df_events['time_obj'].apply(convert_time_to_seconds)

    # Build simulation datetime index
    sim_start_dt = pd.to_datetime(sim_start)
    sim_end_dt = pd.to_datetime(sim_end)
    total_seconds = (sim_end_dt - sim_start_dt).total_seconds()
    n_periods = int(total_seconds // (res_min * 60))
    dt_index = pd.date_range(start=sim_start_dt, periods=n_periods, freq=f'{res_min}min')

    # Precompute simulation helper arrays
    sim_dow = dt_index.dayofweek.to_numpy(dtype=np.int32)
    sim_time_sec = (dt_index.hour * 3600 + dt_index.minute * 60 + dt_index.second).to_numpy(dtype=np.int32)
    # Array of day-of-year for each simulation timestamp:
    sim_doy = np.array([ts.timetuple().tm_yday for ts in dt_index], dtype=np.int32)

    res_sec = res_min * 60  # resolution in seconds
    # Initialize consumption array
    consumption = np.zeros(n_periods, dtype=np.float64)

    # Loop over each event row from the CSV
    for _, row in df_events.iterrows():
        event_day = int(row['day'])
        event_time_sec = int(row['time_sec'])
        prob = float(row['probability'])

        # Find all simulation time indices where (day, time) match the event specification
        mask = (sim_dow == event_day) & (sim_time_sec == event_time_sec)
        matching_indices = np.where(mask)[0]
        if matching_indices.size == 0:
            continue  # no matching simulation times

        # In one vectorized step, decide for all matching indices if an event triggers
        rand_vals = np.random.rand(matching_indices.size)
        triggered_indices = matching_indices[rand_vals <= prob]
        if triggered_indices.size == 0:
            continue

        # For all triggered events, vectorized sampling of duration and flow rate:
        n_triggers = triggered_indices.size
        durations = np.random.normal(row['duration'], row['sigma_duration'], size=n_triggers)
        durations = np.maximum(durations, res_sec)  # ensure at least one time step
        flows = np.random.normal(row['flow_rate'], row['sigma_flow_rate'], size=n_triggers)
        flows = np.maximum(flows, 0.0001)  # prevent non-positive flow

        # Compute event volumes in liters; then convert to m³
        volumes = durations * flows  # liters
        volumes = volumes / 1000.0  # convert to m³

        # Compute seasonal multipliers (vectorized) for these triggered events
        triggered_doy = sim_doy[triggered_indices]
        season_mult = seasonal_vector(triggered_doy, seasonal_amplitude, seasonal_phase)
        volumes *= season_mult

        # Calculate number of steps each event spans, and distribute volume accordingly.
        n_steps_events = np.ceil(durations / res_sec).astype(np.int32)
        # Loop over triggered events (number of events typically is small)
        for i, trig_idx in enumerate(triggered_indices):
            steps = n_steps_events[i]
            start_idx = trig_idx
            end_idx = min(trig_idx + steps, n_periods)
            vol_per_step = volumes[i] / (end_idx - start_idx)
            consumption[start_idx:end_idx] += vol_per_step

    # Compute energy: Energy (J) = Volume (m³) * DENSITY * CP * ΔT, then convert to Wh.
    deltaT = Tw - Tcold
    energy_J = consumption * DENSITY * CP * deltaT
    energy_Wh = energy_J / 3600.0

    consumption_series = pd.Series(consumption, index=dt_index)
    energy_series = pd.Series(energy_Wh, index=dt_index)
    return consumption_series, energy_series


# -------------------------------
# Example usage
# -------------------------------
if __name__ == '__main__':
    start_time = time.perf_counter()
    cons_series, en_series = simulate_dhw_series_vectorized(
        activity_csv=CSV_FILE,
        sim_start=SIM_START_DATE,
        sim_end=SIM_END_DATE,
        res_min=RESOLUTION_MIN,
        Tw=DEFAULT_TW,
        Tcold=DEFAULT_TCOLD,
        seed=RANDOM_SEED,
        seasonal_amplitude=SEASONAL_AMPLITUDE,
        seasonal_phase=SEASONAL_PHASE
    )
    end_time = time.perf_counter()
    print(f"Simulation completed in {end_time - start_time:.6f} seconds")
    print(f"Total water consumption (m³): {cons_series.sum():.3f}")
    print(f"Total energy consumption (Wh): {en_series.sum():.3f}")

    # Optionally save results
    cons_series.to_csv("dhw_consumption_timeseries_vectorized.csv", header=['Consumption_m3'])
    en_series.to_csv("dhw_energy_timeseries_vectorized.csv", header=['Energy_Wh'])
