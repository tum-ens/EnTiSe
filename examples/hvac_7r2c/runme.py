"""
Example script HVAC: 7R2C
The code is identical to the jupyter notebook.
This script demonstrates how to use the 7R2C method to generate indoor temperature time series and heating/cooling
loads based on outdoor weather conditions and building properties.
"""

import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Columns as Cols
from entise.constants import Types
from examples.utils import load_input, run_simulation


def analyze_results(summary: pd.DataFrame, df: dict, data: dict, save_figures: bool = False) -> None:
    def plot_temperature_and_solar_radiation():
        """Figure 1: Indoor & Outdoor Temperature and Solar Radiation (GHI)"""
        fig, ax1 = plt.subplots(figsize=(15, 6))

        # Solar radiation plot (GHI) with separate y-axis
        ax2 = ax1.twinx()
        ax2.plot(
            building_data.index,
            data["weather"][Cols.SOLAR_GHI],
            label="Solar Radiation (GHI)",
            color="tab:orange",
            alpha=0.3,
        )
        ax2.set_ylabel("Solar Radiation (W/m²)")
        ax2.legend(loc="upper right")
        ax2.set_ylim(-250, 1000)

        # Temperature plot
        ax1.plot(
            building_data.index,
            data["weather"][f"{Cols.TEMP_AIR}@2m"],
            label="Outdoor Temp",
            color="tab:cyan",
            alpha=0.7,
        )
        ax1.plot(building_data.index, building_data[Cols.TEMP_IN], label="Indoor Temp", color="tab:blue")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title(f"Building ID: {building_id} - Temperatures and Solar Radiation")
        ax1.legend(loc="upper left")
        ax1.grid(True)
        ax1.set_ylim(-10, 40)

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)  # required to see through ax1 to ax2
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"building_{building_id}_temp_solar.png", dpi=300)
        plt.show()

    def plot_hvac_loads():
        """Figure 2: Heating and Cooling Loads"""
        fig, ax = plt.subplots(figsize=(14, 5))
        heating_MWh = summary.loc[building_id, "heating:demand[Wh]"] / 1e6
        cooling_MWh = summary.loc[building_id, "cooling:demand[Wh]"] / 1e6
        (line1,) = ax.plot(
            building_data.index,
            building_data["heating:load[W]"],
            label=f"Heating: {heating_MWh:.1f} MWh",
            color="tab:red",
            alpha=0.8,
        )
        (line2,) = ax.plot(
            building_data.index,
            building_data["cooling:load[W]"],
            label=f"Cooling: {cooling_MWh:.1f} MWh",
            color="tab:cyan",
            alpha=0.8,
        )
        # Create the combined legend in the upper left corner
        ax.set_ylabel("Load (W)")
        ax.set_title(f"Building ID: {building_id} - HVAC Loads")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"building_{building_id}_hvac_loads.png", dpi=300)
        plt.show()

    def plot_outdoor_temp_with_loads():
        """Figure 3: Outdoor Temperature with Heating & Cooling Loads"""
        fig, ax1 = plt.subplots(figsize=(15, 6))

        # Plot outdoor temperature on left y-axis
        air_temp = data["weather"][f"{Cols.TEMP_AIR}@2m"]
        ax1.plot(building_data.index, air_temp, label="Outdoor Temp", color="tab:cyan", alpha=0.7)

        ax1.set_ylabel("Outdoor Temp (°C)")
        ax1.set_ylim(air_temp.min().round() - 2, air_temp.max().round() + 2)

        # Create second y-axis for loads
        ax2 = ax1.twinx()
        ax2.plot(
            building_data.index, building_data["heating:load[W]"], label="Heating Load", color="tab:red", alpha=0.8
        )
        ax2.plot(
            building_data.index, building_data["cooling:load[W]"], label="Cooling Load", color="tab:blue", alpha=0.8
        )
        ax2.set_ylabel("HVAC Load (W)")
        ax2.set_ylim(
            min(building_data["heating:load[W]"].min(), building_data["cooling:load[W]"].min()) * 1.1,
            max(building_data["heating:load[W]"].max(), building_data["cooling:load[W]"].max()) * 1.1,
        )

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax1.set_title(f"Building ID: {building_id} - Outdoor Temp & HVAC Loads")
        ax1.grid(True)
        fig.tight_layout()
        if save_figures:
            plt.savefig(f"building_{building_id}_temp_hvac.png", dpi=300)
        plt.show()

    # Define the building ID to process and visualize
    building_id = summary.index[0]  # Change index to visualize different buildings

    # Visualize results for the processed building
    # Note: We're using the same building_id as defined above
    building_data = df[building_id][Types.HVAC]

    # Create figures
    plot_temperature_and_solar_radiation()
    plot_hvac_loads()
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
