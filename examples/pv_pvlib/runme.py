"""
Example script PV: pvlib
The code is identical to the jupyter notebook.
This script demonstrates how to use the pvlib method to generate photovoltaic (PV) generation time series
based on weather data and system configurations.
"""

import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import SEP, Types
from entise.constants import Columns as C
from examples.utils import load_input, run_simulation


def analyze_results(df: dict, objects: pd.DataFrame, save_figures: bool = False) -> None:
    def plot_pv_analysis():
        """Figure 1: Comparative analysis between different PV systems"""
        # Get the maximum generation for each system
        max_gen = {}
        total_gen = {}
        for obj_id in df:
            # Extract scalar value from Series
            max_value = df[obj_id][Types.PV].max()
            if hasattr(max_value, "iloc"):
                max_value = max_value.iloc[0]
            max_gen[obj_id] = max_value

            # Extract scalar value from Series
            total_value = df[obj_id][Types.PV].sum() / 1000  # Convert to kWh
            if hasattr(total_value, "iloc"):
                total_value = total_value.iloc[0]
            total_gen[obj_id] = total_value

        # Create a bar chart for maximum generation
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(max_gen)), list(max_gen.values()))
        plt.xticks(range(len(max_gen)), [f"ID {id}" for id in max_gen.keys()], rotation=45)
        plt.title("Maximum PV Generation by System")
        plt.ylabel("Maximum Power (W)")
        plt.grid(axis="y")

        # Create a bar chart for total generation
        plt.subplot(1, 2, 2)
        plt.bar(range(len(total_gen)), list(total_gen.values()))
        plt.xticks(range(len(total_gen)), [f"ID {id}" for id in total_gen.keys()], rotation=45)
        plt.title("Total Annual PV Generation by System")
        plt.ylabel("Total Generation (kWh)")
        plt.grid(axis="y")

        plt.tight_layout()
        if save_figures:
            plt.savefig("pv_comparative_analysis.png", dpi=300)
        plt.show()

    def plot_yearly_timeseries():
        """Figure 2: Year timeseries visualization for all systems"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        # For each PV system, create a separate subplot
        for i, obj_id in enumerate(df):
            # Get azimuth and tilt values for the title
            azimuth = system_configs[obj_id]["azimuth"] if obj_id in system_configs else 0
            tilt = system_configs[obj_id]["tilt"] if obj_id in system_configs else 0

            # Plot the time series
            df[obj_id][Types.PV].plot(ax=axes[i], color="#1f77b4", linewidth=1)
            axes[i].set_title(f"ID {obj_id}, Azimuth: {azimuth}, Tilt: {tilt}")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Power (W)")
            axes[i].grid(True)

        plt.tight_layout()
        if save_figures:
            plt.savefig("pv_yearly_timeseries.png", dpi=300)
        plt.show()

    def plot_monthly_analysis():
        """Figure 3: Monthly generation analysis"""
        # Create a figure with 8 subfigures (2x4 grid)
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
        monthly_data = {}

        # For each PV system, create a separate subplot
        for i, obj_id in enumerate(df):
            ts = df[obj_id][Types.PV]
            # Resample to monthly sums
            monthly = ts.resample("M").sum()
            monthly_data[obj_id] = monthly

            # Get azimuth and tilt values for the title
            azimuth = system_configs[obj_id]["azimuth"] if obj_id in system_configs else 0
            tilt = system_configs[obj_id]["tilt"] if obj_id in system_configs else 0

            # Plot on the corresponding subplot
            axes[i].plot(monthly.index, monthly.values / 1000)
            axes[i].set_title(f"ID {obj_id}, Azimuth: {azimuth}, Tilt: {tilt}")
            axes[i].set_xlabel("Month")
            axes[i].set_ylabel("Generation (kWh)")
            axes[i].grid(True)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save_figures:
            plt.savefig("pv_monthly_analysis.png", dpi=300)
        plt.show()

    def plot_seasonal_daily_profile():
        """Figure 4: Seasonal daily profile analysis"""
        # Define seasons
        seasons = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}

        # Create a figure with 8 subfigures (2x4 grid)
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        # For each PV system, create a separate subplot
        for i, obj_id in enumerate(df):
            ts = df[obj_id][Types.PV]

            # Get azimuth and tilt values for the title
            azimuth = system_configs[obj_id]["azimuth"] if obj_id in system_configs else 0
            tilt = system_configs[obj_id]["tilt"] if obj_id in system_configs else 0

            # Plot each season on the same subplot
            for season_name, months in seasons.items():
                # Filter data for the season
                season_data = ts[ts.index.month.isin(months)]
                # Create average daily profile
                daily_profile = season_data.groupby(season_data.index.hour).mean()
                axes[i].plot(daily_profile.index, daily_profile.values, label=season_name, linewidth=2)

            axes[i].set_title(f"ID {obj_id}, Azimuth: {azimuth}, Tilt: {tilt}")
            axes[i].set_xlabel("Hour of Day")
            axes[i].set_ylabel("Average Generation (W)")
            axes[i].legend()
            axes[i].grid(True)
            axes[i].set_xticks(range(0, 24, 4))  # Show fewer ticks for readability

        plt.tight_layout()
        plt.show()

    def plot_daily_heatmap():
        """Figure 5: Heatmap of daily generation patterns"""
        # Create a figure with 8 subfigures (2x4 grid)
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        # For each PV system, create a separate subplot
        for i, obj_id in enumerate(df):
            ts = df[obj_id][Types.PV]

            # Get azimuth and tilt values for the title
            azimuth = system_configs[obj_id]["azimuth"] if obj_id in system_configs else 0
            tilt = system_configs[obj_id]["tilt"] if obj_id in system_configs else 0

            # Create a pivot table with hours as columns and days as rows
            daily_data = ts.copy()
            daily_data.index = pd.MultiIndex.from_arrays(
                [daily_data.index.date, daily_data.index.hour], names=["date", "hour"]
            )
            daily_pivot = daily_data.unstack(level="hour")

            # Create heatmap
            im = axes[i].imshow(daily_pivot, aspect="auto", cmap="viridis")
            axes[i].set_title(f"ID {obj_id}, Azimuth: {azimuth}, Tilt: {tilt}")
            axes[i].set_xlabel("Hour of Day")
            axes[i].set_ylabel("Day of Year")
            axes[i].set_xticks(range(0, 24, 6))  # Show fewer ticks for readability

            # Add colorbar to each subplot
            fig.colorbar(im, ax=axes[i], shrink=0.7, label="Generation (W)")

        plt.tight_layout()
        if save_figures:
            plt.savefig("pv_daily_heatmap.png", dpi=300)
        plt.show()

    # Convert index to datetime for all time series
    for obj_id in df:
        df[obj_id][Types.PV].index = pd.to_datetime(df[obj_id][Types.PV].index)

    # Get azimuth and tilt values from objects dataframe
    system_configs = {}
    for _, row in objects.iterrows():
        obj_id = row["id"]
        if obj_id in df:
            azimuth = row["azimuth[degree]"] if not pd.isna(row["azimuth[degree]"]) else 0
            tilt = row["tilt[degree]"] if not pd.isna(row["tilt[degree]"]) else 0
            power = row["power[W]"] if not pd.isna(row["power[W]"]) else 1
            system_configs[obj_id] = {"azimuth": azimuth, "tilt": tilt, "power": power}

    # Generate all plots
    plot_pv_analysis()
    plot_yearly_timeseries()
    plot_monthly_analysis()
    plot_seasonal_daily_profile()
    plot_daily_heatmap()


def main(print_summary: bool = False, analysis: bool = False, save_figures=False) -> None:
    objects, data = load_input()
    summary, df = run_simulation(objects, data, workers=1)
    if print_summary:
        print("Summary:")
        summary_kwh = summary
        summary_kwh[f"{Types.PV}{SEP}{C.GENERATION}"] /= 1000
        summary_kwh.rename(columns=lambda x: x.replace("[Wh]", "[kWh]"), inplace=True)
        summary_kwh = summary_kwh.round(0).astype(int)
        print(summary_kwh)
    if analysis:
        analyze_results(df, objects, save_figures)


if __name__ == "__main__":
    main(print_summary=True, analysis=True, save_figures=False)
