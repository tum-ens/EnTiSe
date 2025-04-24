import numpy as np
import pandas as pd
import time


class DHWTimeSeriesCalculator:
    def __init__(self, start_date: str, end_date: str, resolution_min: int, daily_demand: float,
                 Tw: float = 60, Tcold: float = 10, seed: int = None,
                 event_mean_duration: float = 5, event_duration_std: float = 1,
                 event_flow_rate: float = 0.01, event_flow_std: float = 0.002):
        """
        Initialize the DHW time series calculator.

        Parameters:
            start_date (str): Simulation start date (e.g., '2020-01-01').
            end_date (str): Simulation end date (non-inclusive, e.g., '2020-01-03').
            resolution_min (int): Time resolution in minutes.
            daily_demand (float): Mean daily DHW consumption [m³].
            Tw (float): Outlet temperature (°C).
            Tcold (float): Inlet (cold) temperature (°C).
            seed (int): Optional seed for reproducibility.
            event_mean_duration (float): Mean duration for each tapping event (in minutes).
            event_duration_std (float): Standard deviation for event duration.
            event_flow_rate (float): Mean flow rate for an event (in m³/min).
            event_flow_std (float): Standard deviation for the flow rate.
        """
        if seed is not None:
            np.random.seed(seed)

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.resolution_min = resolution_min
        self.daily_demand = daily_demand
        self.Tw = Tw
        self.Tcold = Tcold

        # Physical constants
        self.rho = 1000  # kg/m³
        self.cp = 4186  # J/(kg·°C)

        # Tapping event parameters
        self.event_mean_duration = event_mean_duration
        self.event_duration_std = event_duration_std
        self.event_flow_rate = event_flow_rate
        self.event_flow_std = event_flow_std

        # Build datetime index
        total_seconds = (self.end_date - self.start_date).total_seconds()
        periods = int(total_seconds // (resolution_min * 60))
        self.dt_index = pd.date_range(start=self.start_date, periods=periods, freq=f'{resolution_min}min')

    def generate_timeseries(self):
        """
        Generate the DHW consumption and energy time series.

        Returns:
            consumption_series (pd.Series): Consumption per time step (m³).
            energy_series (pd.Series): Energy required per time step (Wh).
        """
        # Vectorized extraction of unique days
        day_labels = self.dt_index.normalize()  # Each timestamp converted to its date
        unique_days = np.unique(day_labels)

        n_steps_day = int(24 * 60 / self.resolution_min)
        daily_consumptions = []

        for day in unique_days:
            daily_consumption = self._simulate_day(n_steps_day)
            daily_consumptions.append(daily_consumption)

        # Concatenate daily time series arrays
        consumption_array = np.concatenate(daily_consumptions)
        consumption_series = pd.Series(consumption_array, index=self.dt_index)

        # Compute energy for each time step in Wh
        deltaT = self.Tw - self.Tcold
        # Energy in Joules = volume * density * cp * deltaT
        # Wh = J / 3600
        energy_J = consumption_series.values * self.rho * self.cp * deltaT
        energy_Wh = energy_J / 3600

        energy_series = pd.Series(energy_Wh, index=self.dt_index)
        return consumption_series, energy_series

    def _simulate_day(self, n_steps_day: int):
        """
        Simulate a single day of DHW consumption.

        Parameters:
            n_steps_day (int): Number of time steps in the day.

        Returns:
            daily_consumption (np.ndarray): Array of consumption values per time step (m³).
        """
        target_volume = self.daily_demand  # m³

        # Estimate average event volume given mean duration and flow rate
        avg_event_vol = self.event_flow_rate * self.event_mean_duration
        n_events = max(1, int(np.ceil(target_volume / avg_event_vol)))

        # Random event start indices within the day's time steps
        event_starts = np.sort(np.random.uniform(0, n_steps_day, n_events).astype(int))

        # Generate event durations in minutes, then convert to time steps
        durations = self.event_mean_duration + self.event_duration_std * np.random.randn(n_events)
        durations = np.clip(durations, self.resolution_min, None)
        steps_per_event = np.ceil(durations / self.resolution_min).astype(int)

        # Generate event flow rates (m³/min)
        flow_rates = self.event_flow_rate + self.event_flow_std * np.random.randn(n_events)
        flow_rates = np.clip(flow_rates, 0.0001, None)

        # Calculate event volumes and scale to meet target volume exactly
        event_volumes = flow_rates * durations  # in m³
        scaling_factor = target_volume / event_volumes.sum()
        event_volumes *= scaling_factor

        # Allocate the event volumes uniformly over their duration.
        # Instead of iterating, we gather the indices and use np.add.at
        day_array = np.zeros(n_steps_day)
        for start, steps, vol in zip(event_starts, steps_per_event, event_volumes):
            end = min(start + steps, n_steps_day)
            actual_steps = end - start
            day_array[start:end] += vol / actual_steps

        # Correct for rounding errors: adjust final time step.
        day_array[-1] += target_volume - day_array.sum()
        return day_array


# ----------------------------------------------------
# Example usage
# ----------------------------------------------------
if __name__ == '__main__':
    start_time = time.perf_counter()

    calc = DHWTimeSeriesCalculator(
        start_date='2020-01-01',
        end_date='2020-12-31',  # 3-day simulation
        resolution_min=15,
        daily_demand=0.2,  # m³ per day
        Tw=60,
        Tcold=10,
        seed=42,
        event_mean_duration=5,
        event_duration_std=1,
        event_flow_rate=0.01,
        event_flow_std=0.002
    )

    consumption_series, energy_series = calc.generate_timeseries()

    end_time = time.perf_counter()
    print(f"Simulation completed in {end_time - start_time:.6f} seconds")
    print("Total water consumption (m³):", consumption_series.sum())
    print("Total energy consumption (Wh):", energy_series.sum())
