"""
Example script DHW: JordanVajen
The code is identical to the jupyter notebook.
This script demonstrates how to use the probabilistic DHW method by Jordan et. al. to generate
domestic hot water demand time series for multiple buildings based on dwelling size.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Objects as O
from entise.constants import Types
from examples.utils import load_input, run_simulation


def analyze_results(df: dict, objects: pd.DataFrame, save_figures: bool = False) -> None:
    def plot_dhw_demands():
        def plot_dhw_demand(ax, obj_id, df, title):
            obj_data = df[obj_id][Types.DHW]
            obj_data.index = pd.to_datetime(obj_data.index)

            # Plot volume
            ax.plot(obj_data.index, obj_data[f"{Types.DHW}:volume[l]"], label="DHW Volume (liters)", color="blue")
            ax.set_ylabel("Volume (liters)")
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.grid(True)

            # Add second y-axis for energy
            ax_energy = ax.twinx()
            ax_energy.plot(
                obj_data.index, obj_data[f"{Types.DHW}:energy[Wh]"], label="DHW Energy (Wh)", color="red", alpha=0.7
            )
            ax_energy.set_ylabel("Energy (Wh)")
            ax_energy.legend(loc="upper right")

            # Format x-axis for better readability
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days

            return obj_data

        # Create a 4x2 grid of plots for all 8 objects
        fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
        axes = axes.flatten()

        # Plot each object
        objects.set_index("id", inplace=True)
        for i, (obj_id, object) in enumerate(objects.iterrows()):
            plot_dhw_demand(axes[i], obj_id, df, f"Building ID: {obj_id} - Dwelling Size: {object[O.DWELLING_SIZE]}")

        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.tight_layout()
        plt.show()

    # Function to plot daily profile
    def plot_daily_profiles():
        def plot_daily_profile(ax, obj_data, title):
            obj_data["hour"] = obj_data.index.hour
            daily_profile = obj_data.groupby("hour")[f"{Types.DHW}:volume[l]"].mean()
            ax.bar(daily_profile.index, daily_profile.values, color="skyblue", alpha=0.7)
            ax.set_xlabel("Hour of Day")
            ax.set_ylabel("Average DHW Volume (liters)")
            ax.set_title(title)
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)

            # Calculate yearly total in m3
            yearly_total = obj_data[f"{Types.DHW}:volume[l]"].sum() / 1000  # Convert L to m3
            ax.text(
                0.95,
                0.95,
                f"Yearly total: {yearly_total:.1f} m³",
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Create a 4x2 grid for daily profiles
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.flatten()

        # Plot daily profiles for each object
        for i, (obj_id, object) in enumerate(objects.iterrows()):
            plot_daily_profile(
                axes[i], df[obj_id][Types.DHW], f"Building ID: {obj_id} - Dwelling Size: {object[O.DWELLING_SIZE]}"
            )

        plt.tight_layout()
        if save_figures:
            plt.savefig("daily_profiles.png", dpi=300)
        plt.show()

        # Compare weekend vs. weekday profiles for object 6 (with weekend activity)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract weekday and weekend data
        obj6_data = df[6][Types.DHW]
        obj6_data["day_of_week"] = obj6_data.index.dayofweek
        weekday_data = obj6_data[obj6_data["day_of_week"] < 5]  # Monday-Friday
        weekend_data = obj6_data[obj6_data["day_of_week"] >= 5]  # Saturday-Sunday

        # Calculate average profiles
        weekday_profile = weekday_data.groupby("hour")[f"{Types.DHW}:volume[l]"].mean()
        weekend_profile = weekend_data.groupby("hour")[f"{Types.DHW}:volume[l]"].mean()

        # Plot both profiles
        ax.bar(weekday_profile.index - 0.2, weekday_profile.values, width=0.4, color="blue", alpha=0.7, label="Weekday")
        ax.bar(weekend_profile.index + 0.2, weekend_profile.values, width=0.4, color="red", alpha=0.7, label="Weekend")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average DHW Volume (liters)")
        ax.set_title("Building ID: 6 - Dwelling Size: 80 m² - Weekday vs. Weekend DHW Profiles")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save_figures:
            plt.savefig("object6_weekday_weekend_profile.png", dpi=300)
        plt.show()

    plot_dhw_demands()
    plot_daily_profiles()


def main(print_summary: bool = False, analysis: bool = False, save_figures=False) -> None:
    objects, data = load_input()
    summary, df = run_simulation(objects, data, workers=1)
    if print_summary:
        print("\nSummary:")
        print(summary.to_string())
    if analysis:
        analyze_results(df, objects, save_figures)


if __name__ == "__main__":
    main(print_summary=True, analysis=True, save_figures=False)
