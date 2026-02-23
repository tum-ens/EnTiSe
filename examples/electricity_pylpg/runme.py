"""
Example script Electricity: PyLPG

Standardized to match the Demandlib electricity runme pattern:
- Uses examples.utils.load_input() + examples.utils.run_simulation()
- Optional summary print in kWh/kW
- Optional analysis plotting (weekly + seasonal average daily profiles)
- Optional CSV export to ./output/

Notes:
- PyLPG method internally uses weather[C.DATETIME] as horizon.
- This runme does NOT create run_1/run_2 folders; it mirrors demandlib's simple pattern.
"""


import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Types
from examples.utils import load_input, run_simulation


def analyze_results(results: dict, objects: pd.DataFrame, save_figures: bool = False) -> None:
    def plot_weekly_profile(series: pd.Series, building_id: str):
        # Pick the first full week starting Monday (or fallback: first 7 days)
        start = series.index.min().normalize()
        start = start + pd.Timedelta(days=(7 - start.weekday()) % 7)
        end = start + pd.Timedelta(days=7)

        s_week = series.loc[(series.index >= start) & (series.index < end)]
        if s_week.empty:
            start = series.index.min()
            end = start + pd.Timedelta(days=7)
            s_week = series.loc[(series.index >= start) & (series.index < end)]

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(s_week.index, s_week.values, alpha=0.9)
        ax.set_title(f"Building ID: {building_id} - Electricity Load (Weekly)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Electricity load (W)")
        ax.grid(True)
        plt.tight_layout()
        if save_figures:
            plt.savefig(f"building_id_weekly_{building_id}.png", dpi=150)
        plt.show()

    def plot_seasonal_subplots(series: pd.Series, building_id: str):
        df_tmp = series.to_frame("load")
        # Ensure the index is a DatetimeIndex
        if not isinstance(df_tmp.index, pd.DatetimeIndex):
            df_tmp.index = pd.to_datetime(df_tmp.index, utc=True)
        df_tmp["month"] = df_tmp.index.month
        df_tmp["hour"] = df_tmp.index.hour

        # Map month → season
        df_tmp["season"] = "Autumn (SON)"
        df_tmp.loc[df_tmp["month"].isin([12, 1, 2]), "season"] = "Winter (DJF)"
        df_tmp.loc[df_tmp["month"].isin([3, 4, 5]), "season"] = "Spring (MAM)"
        df_tmp.loc[df_tmp["month"].isin([6, 7, 8]), "season"] = "Summer (JJA)"
        df_tmp.loc[df_tmp["month"].isin([9, 10, 11]), "season"] = "Autumn (SON)"

        prof = df_tmp.groupby(["season", "hour"], as_index=False)["load"].mean()

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
            plt.savefig(f"seasonal_average_daily_{building_id}.png", dpi=150)
        plt.show()

    # Pick first object for analysis (same as demandlib runme)
    building_id = objects["id"].iloc[0]
    res = results.get(building_id)
    if res is None:
        raise KeyError(f"[pylpg runme] No results found for object id={building_id}")

    ts = res[Types.ELECTRICITY]
    if isinstance(ts, dict):
        ts = next(iter(ts.values()))

    # If single column -> use it, else sum columns (defensive fallback)
    if ts.shape[1] == 1:
        series = ts.iloc[:, 0].copy()
    else:
        series = ts.sum(axis=1).copy()

    series.index = pd.to_datetime(series.index)

    plot_weekly_profile(series, str(building_id))
    plot_seasonal_subplots(series, str(building_id))


def main(num: int = 1, print_summary: bool = False, analysis: bool = False, save_figures: bool = False) -> None:
    objects, data = load_input()
    if num == 1:
        print(
            "Only the first element is simulated due to the extremly long simulation times."
            'If you want to run more increase the "num" parameter.'
        )
    objects = objects.iloc[: min(num, len(objects))]
    summary, results = run_simulation(objects, data, workers=1)
    if print_summary:
        print("Summary:")
        summary_kwh = (summary / 1000).round(1)
        summary_kwh.rename(columns=lambda x: x.replace("[W]", "[kW]").replace("[Wh]", "[kWh]"), inplace=True)
        print(summary_kwh.to_string())

    if analysis:
        analyze_results(results, objects, save_figures=save_figures)


if __name__ == "__main__":
    main(num=1, print_summary=True, analysis=True, save_figures=False)
