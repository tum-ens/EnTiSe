#!/usr/bin/env python
import numpy as np
import pandas as pd
import datetime as dt
import os

# -------------------------------
# Settings and constants
# -------------------------------
SIM_YEAR = 2019
RESOLUTION_MIN = 15  # time resolution in minutes

# DHW energy parameters (if needed)
DEFAULT_TW = 60  # Outlet temperature in °C
DEFAULT_TCOLD = 10  # Inlet temperature in °C
DENSITY = 1000  # kg/m³
CP = 4186  # J/(kg·°C)

# Seasonal factor parameters (fractional variation)
SEASONAL_AMPLITUDE = 0.1
SEASONAL_PHASE = 1  # Day-of-year when multiplier is maximum

# CSV filename (weekly event schedule)
CSV_FILE = 'dhw_activity_adjusted.csv'
RANDOM_SEED = 42


# -------------------------------
# Utility functions
# -------------------------------
def parse_time_str(time_str: str) -> dt.time:
    """Parse a HH:MM:SS string to a time object."""
    return dt.datetime.strptime(time_str, '%H:%M:%S').time()


def convert_time_to_minutes(t: dt.time) -> int:
    """Convert a time object to minutes since midnight."""
    return t.hour * 60 + t.minute  # ignore seconds for matching


def seasonal_multiplier(ts: pd.Timestamp, amplitude: float, phase: float) -> float:
    """
    Compute seasonal multiplier for a given timestamp.
    Uses a cosine: multiplier = 1 + amplitude * cos(2*pi*(doy - phase)/365)
    """
    doy = ts.timetuple().tm_yday
    return 1 + amplitude * np.cos(2 * np.pi * (doy - phase) / 365)


# -------------------------------
# Main simulation function
# -------------------------------
def simulate_yearly_events(activity_csv: str,
                           year: int,
                           res_min: int,
                           seed: int = None,
                           seasonal_amplitude: float = SEASONAL_AMPLITUDE,
                           seasonal_phase: float = SEASONAL_PHASE) -> pd.DataFrame:
    """
    Convert a weekly event schedule CSV into a yearly time series (for 2019) for each event type.

    The CSV is assumed to have columns:
       - day: integer 0 (Monday) to 6 (Sunday)
       - time: time of day in HH:MM:SS format
       - event: event type (string)
       - probability: probability the event occurs at its scheduled time
       - duration: mean duration (seconds)
       - sigma_duration: stdev of duration (seconds)
       - flow_rate: mean flow rate (liters/second)
       - sigma_flow_rate: stdev of flow rate (liters/second)

    The function converts durations to minutes and flow rates to liters per minute.
    For each simulation timestep (at the chosen resolution) within the year,
    the method checks if an event is scheduled based on the day-of-week and the time.
    If the event triggers (based on its probability), an event volume is computed as:
         volume (L) = duration (s) * flow_rate (L/s)
    The volume is then scaled by a seasonal multiplier and converted to m³.
    The event volume is uniformly distributed over the number of timesteps spanned.

    Returns:
         A DataFrame with index equal to simulation timestamps for the full year,
         and one column per unique event type containing the water consumption (m³)
         from that event type.
    """
    if seed is not None:
        np.random.seed(seed)

    # Load the weekly event definitions
    if not os.path.exists(activity_csv):
        raise FileNotFoundError(f"CSV file {activity_csv} not found.")
    df_events = pd.read_csv(activity_csv)

    # Convert the time column to time objects, then to minutes since midnight
    df_events['time_obj'] = df_events['time'].apply(parse_time_str)
    df_events['time_min'] = df_events['time_obj'].apply(convert_time_to_minutes)

    # Convert duration from seconds to minutes; convert flow rate from L/s to L/min.
    df_events['duration_min'] = df_events['duration'] / 60.0
    df_events['sigma_duration_min'] = df_events['sigma_duration'] / 60.0
    df_events['flow_rate_lpm'] = df_events['flow_rate'] * 60.0
    df_events['sigma_flow_rate_lpm'] = df_events['sigma_flow_rate'] * 60.0

    # Define simulation period for the given year (non-inclusive end)
    sim_start = f'{year}-01-01'
    sim_end = f'{year + 1}-01-01'
    sim_start_dt = pd.to_datetime(sim_start)
    sim_end_dt = pd.to_datetime(sim_end)

    total_seconds = (sim_end_dt - sim_start_dt).total_seconds()
    n_steps = int(total_seconds // (res_min * 60))

    # Create simulation datetime index using periods (this avoids the 'closed' parameter)
    dt_index = pd.date_range(start=sim_start_dt, periods=n_steps, freq=f'{res_min}min')

    # Precompute helper arrays for simulation timesteps
    sim_day_of_week = dt_index.dayofweek.to_numpy(dtype=np.int32)  # Monday=0
    sim_time_min = (dt_index.hour * 60 + dt_index.minute).to_numpy(dtype=np.int32)

    # Prepare an empty dictionary to accumulate consumption for each event type.
    event_types = df_events['event'].unique()
    consumption_dict = {etype: np.zeros(n_steps, dtype=np.float64) for etype in event_types}

    # For each event definition in the weekly schedule:
    for _, row in df_events.iterrows():
        event_day = int(row['day'])
        event_time_min = int(row['time_min'])
        prob = float(row['probability'])
        mean_duration = float(row['duration_min'])
        sigma_duration = float(row['sigma_duration_min'])
        mean_flow = float(row['flow_rate_lpm'])
        sigma_flow = float(row['sigma_flow_rate_lpm'])

        # Find simulation indices that match the scheduled (day, time)
        mask = (sim_day_of_week == event_day) & (sim_time_min == event_time_min)
        matching_idx = np.where(mask)[0]
        if matching_idx.size == 0:
            continue

        # For all matching timesteps, decide which trigger an event.
        rand_vals = np.random.rand(matching_idx.size)
        triggers = matching_idx[rand_vals <= prob]
        if triggers.size == 0:
            continue

        # Sample event duration and flow rate for each triggered occurrence
        durations = np.random.normal(mean_duration, sigma_duration, size=triggers.size)
        durations = np.maximum(durations, res_min)  # at least one time step (in minutes)
        flows = np.random.normal(mean_flow, sigma_flow, size=triggers.size)
        flows = np.maximum(flows, 0.0001)

        # Compute event volumes in liters; convert duration from minutes to seconds for calculation
        volumes_liters = (durations * 60) * flows
        volumes_m3 = volumes_liters / 1000.0

        # Compute seasonal multipliers for triggered times
        triggered_times = dt_index[triggers]
        season_mult = np.array([seasonal_multiplier(ts, seasonal_amplitude, seasonal_phase) for ts in triggered_times])
        volumes_m3 *= season_mult

        # Determine number of steps each event spans and distribute volumes
        n_steps_events = np.ceil(durations / res_min).astype(np.int32)
        for i, start_idx in enumerate(triggers):
            steps = n_steps_events[i]
            end_idx = min(start_idx + steps, n_steps)
            consumption_dict[row['event']][start_idx:end_idx] += volumes_m3[i] / (end_idx - start_idx)

    # Combine consumption per event type into a single DataFrame
    df_consumption = pd.DataFrame({etype: consumption_dict[etype] for etype in event_types}, index=dt_index)
    df_consumption['Total_m3'] = df_consumption.sum(axis=1)

    # Energy calculation (if needed)
    deltaT = DEFAULT_TW - DEFAULT_TCOLD
    energy_J = df_consumption['Total_m3'] * DENSITY * CP * deltaT
    energy_Wh = energy_J / 3600.0
    df_consumption['Energy_Wh'] = energy_Wh

    return df_consumption


# -------------------------------
# Example usage
# -------------------------------
if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    start_time = dt.datetime.now()
    df_yearly = simulate_yearly_events(
        activity_csv=CSV_FILE,
        year=2019,
        res_min=RESOLUTION_MIN,
        seed=RANDOM_SEED,
        seasonal_amplitude=SEASONAL_AMPLITUDE,
        seasonal_phase=SEASONAL_PHASE
    )
    end_time = dt.datetime.now()

    print(f"Simulation completed in {(end_time - start_time).total_seconds():.3f} seconds")
    print("Total yearly water consumption (m³) per event type and overall:")
    print(df_yearly[['Total_m3', 'Energy_Wh']].sum())

    # Optionally, save to CSV
    df_yearly.to_csv("yearly_dhw_timeseries_2019.csv")
