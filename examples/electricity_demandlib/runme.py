"""
Example script Electricity: Demandlib
The code is identical to the jupyter notebook.
This script demonstrates how to use the Demandlib method to generate electricity demand time series based on
household type and demand.
"""

import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Types
from examples.utils import load_input, run_simulation


def analyze_results(df: dict, objects: pd.DataFrame, save_figures: bool = False) -> None:
    def plot_weekly_profile():
        # Pick the first full week starting Monday (or fallback: first 7 days)
        start = df.index.min().normalize()
        start = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
        end = start + pd.Timedelta(days=7)

        s_week = df.loc[(df.index >= start) & (df.index < end)]

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(s_week.index, s_week.values, alpha=0.9)
        ax.set_title(f"Building ID: {building_id} - Electricity Load (Weekly)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Electricity load (W)")
        ax.grid(True)
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"building_id_weekly_{building_id}.png")
        plt.show()

    def plot_seasonal_subplots():
        df_tmp = df.copy()
        df_tmp.columns = ["load"]
        df_tmp["month"] = df_tmp.index.month
        df_tmp["hour"] = df_tmp.index.hour

        # Map month → season (no helper defs)
        df_tmp["season"] = "Autumn (SON)"
        df_tmp.loc[df_tmp["month"].isin([12, 1, 2]), "season"] = "Winter (DJF)"
        df_tmp.loc[df_tmp["month"].isin([3, 4, 5]), "season"] = "Spring (MAM)"
        df_tmp.loc[df_tmp["month"].isin([6, 7, 8]), "season"] = "Summer (JJA)"
        df_tmp.loc[df_tmp["month"].isin([9, 10, 11]), "season"] = "Autumn (SON)"

        prof = df_tmp.groupby(["season", "hour"])["load"].mean().reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        season_order = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Autumn (SON)"]

        for i, season in enumerate(season_order):
            ax = axes[i]
            df_season = prof[prof["season"] == season]
            ax.plot(df_season["hour"], df_season["load"], alpha=0.9)
            ax.set_title(season)
            ax.grid(True)

        fig.suptitle(f"Building ID: {building_id} - Seasonal Average Daily Profiles", fontsize=14)
        fig.supxlabel("Hour of day")
        fig.supylabel("Electricity load (W)")
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"seasonal_average_daily_{building_id}.png")
        plt.show()

    # Prepare data
    building_id = objects["id"].iloc[0]
    res = df.get(building_id)

    df = res[Types.ELECTRICITY]
    df.index = pd.to_datetime(df.index, utc=True)

    # Create plots
    plot_weekly_profile()
    plot_seasonal_subplots()


def main(print_summary: bool = False, analysis: bool = False, save_figures=False) -> None:
    objects, data = load_input()
    summary, df = run_simulation(objects, data, workers=1)
    if print_summary:
        print("Summary:")
        summary_kwh = (summary / 1000).round(0).astype(int)
        summary_kwh.rename(columns=lambda x: x.replace("[W]", "[kW]").replace("[Wh]", "[kWh]"), inplace=True)
        print(summary_kwh.to_string())
    if analysis:
        analyze_results(df, objects, save_figures)


if __name__ == "__main__":
    main(print_summary=True, analysis=True, save_figures=False)
