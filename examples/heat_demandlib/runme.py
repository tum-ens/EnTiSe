"""
Example script Heating: demandlib
The code is identical to the jupyter notebook.
This script demonstrates how to use the demandlib method to generate heating demand time series based on
demand and weather data.
"""

import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Columns as Cols
from entise.constants import Types
from examples.utils import load_input, run_simulation


def analyze_results(summary: pd.DataFrame, df: dict, data: dict, save_figures: bool = False) -> None:
    def plot_heating_loads():
        """Heating and Cooling Loads"""
        fig, ax = plt.subplots(figsize=(14, 5))
        heating_MWh = summary.loc[building_id, "heating:demand[Wh]"] / 1e6
        (line1,) = ax.plot(
            pd.to_datetime(building_data.index, utc=True),
            building_data["heating:load[W]"],
            label=f"Heating: {heating_MWh:.1f} MWh",
            color="tab:red",
            alpha=0.8,
        )
        # Create the combined legend in the upper left corner
        ax.set_ylabel("Load (W)")
        ax.set_title(f"Building ID: {building_id} - Heating Loads")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"building_{building_id}_heating_loads.png", dpi=300)
        plt.show()

    def plot_outdoor_temp_with_loads():
        """Figure 3: Outdoor Temperature with Heating & Cooling Loads"""
        fig, ax1 = plt.subplots(figsize=(15, 6))

        # Plot outdoor temperature on left y-axis
        air_temp = data["weather"][f"{Cols.TEMP_AIR}@2m"]
        ax1.plot(
            pd.to_datetime(building_data.index, utc=True), air_temp, label="Outdoor Temp", color="tab:cyan", alpha=0.7
        )

        ax1.set_ylabel("Outdoor Temp (°C)")
        ax1.set_ylim(air_temp.min().round() - 2, air_temp.max().round() + 2)

        # Create second y-axis for loads
        ax2 = ax1.twinx()
        ax2.plot(
            pd.to_datetime(building_data.index, utc=True),
            building_data["heating:load[W]"],
            label="Heating Load",
            color="tab:red",
            alpha=0.8,
        )
        ax2.set_ylabel("HVAC Load (W)")
        ax2.set_ylim(
            building_data["heating:load[W]"].min() * 1.1,
            building_data["heating:load[W]"].max() * 1.1,
        )

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax1.set_title(f"Building ID: {building_id} - Outdoor Temp & Heating Loads")
        ax1.grid(True)
        fig.tight_layout()
        if save_figures:
            plt.savefig(f"building_{building_id}_temp_heating.png", dpi=300)
        plt.show()

    # Define the building ID to process and visualize
    building_id = summary.index[0]  # Change index to visualize different buildings

    # Visualize results for the processed building
    # Note: We're using the same building_id as defined above
    building_data = df[building_id][Types.HEATING]

    # Create figures
    plot_heating_loads()
    plot_outdoor_temp_with_loads()


def main(print_summary: bool = False, analysis: bool = False, save_figures=False) -> None:
    objects, data = load_input()
    summary, df = run_simulation(objects, data, workers=1)
    if print_summary:
        print("Summary:")
        summary_kwh = (summary / 1000).round(0).astype(int)
        summary_kwh.rename(columns=lambda x: x.replace("[W]", "[kW]").replace("[Wh]", "[kWh]"), inplace=True)
        print(summary_kwh.to_string())
    if analysis:
        analyze_results(summary, df, data, save_figures)


if __name__ == "__main__":
    main(print_summary=True, analysis=True, save_figures=False)
