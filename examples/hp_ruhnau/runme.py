"""
Example script Heat Pump: Ruhnau
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from entise.constants import SEP, Types
from entise.core.generator import TimeSeriesGenerator


def get_input(path: str):
    """
    Benchmark-compatible loader.

    Returns
    -------
    objects : pd.DataFrame
    data : dict
    """
    objects = pd.read_csv(os.path.join(path, "objects.csv"))

    data = {}
    data_folder = os.path.join(path, "data")
    common_data_folder = os.path.join(path, "../common_data")

    # common_data
    if os.path.isdir(common_data_folder):
        for file in os.listdir(common_data_folder):
            fp = os.path.join(common_data_folder, file)
            if file.endswith(".csv"):
                name = file.split(".")[0]
                data[name] = pd.read_csv(fp, parse_dates=True)

    # data/
    if os.path.isdir(data_folder):
        for file in os.listdir(data_folder):
            fp = os.path.join(data_folder, file)
            if file.endswith(".csv"):
                name = file.split(".")[0]
                data[name] = pd.read_csv(fp, parse_dates=True)
            elif file.endswith(".json"):
                name = file.split(".")[0]
                with open(fp, "r") as f:
                    data[name] = json.load(f)

    return objects, data


def simulate(
    objects: pd.DataFrame,
    data: dict,
    workers: int,
    path: str,
    export: bool = True,
    plot: bool = True,
):
    """
    Benchmark-compatible simulator.

    - For benchmarking: call with export=False, plot=False (fast: only gen.generate()).
    - For normal usage: export/plot True (same behavior as supervisor script).

    Returns
    -------
    summary, df
    """
    # Instantiate and configure the generator
    gen = TimeSeriesGenerator()
    gen.add_objects(objects)

    # Generate time series (THIS is what benchmarking wants to time)
    summary, df = gen.generate(data, workers=workers)

    # Benchmark fast path: absolutely nothing else
    if not (export or plot):
        return summary, df

    # ------------------------------------------------------------
    # Everything below is your existing "script logic"
    # (kept intact, only moved inside this function)
    # ------------------------------------------------------------

    print("Loaded data keys:", list(data.keys()))

    # Print summary
    print("Summary:")
    print(summary.to_string())

    # Convert index to datetime for all time series
    for obj_id in df:
        if Types.HP in df[obj_id]:
            df[obj_id][Types.HP].index = pd.to_datetime(df[obj_id][Types.HP].index, utc=True)

    # Get heat pump parameters from objects dataframe
    system_configs = {}
    for _, row in objects.iterrows():
        obj_id = row["id"]
        if obj_id in df:
            hp_source = row["hp_source"] if not pd.isna(row.get("hp_source", pd.NA)) else "Default"
            hp_sink = row["hp_sink"] if not pd.isna(row.get("hp_sink", pd.NA)) else "Default"
            temp_sink = row["temp_sink"] if not pd.isna(row.get("temp_sink", pd.NA)) else "Default"
            temp_water = row["temp_water"] if not pd.isna(row.get("temp_water", pd.NA)) else "Default"
            system_configs[obj_id] = {
                "hp_source": hp_source,
                "hp_sink": hp_sink,
                "temp_sink": temp_sink,
                "temp_water": temp_water,
            }

    # If user only wants export (no plot), stop here (keeps behavior flexible)
    if not plot:
        return summary, df

    # ------------------------------------------------------------
    # PLOTS (your original code, unchanged — only indented)
    # ------------------------------------------------------------

    # Figure 1: Comparative analysis between different heat pump systems
    # Collect the full distribution of COP values for each system
    heating_cop_data = []
    dhw_cop_data = []
    system_ids = []

    for obj_id in df:
        if Types.HP in df[obj_id]:
            # Extract heating COP
            heating_col = f"{Types.HP}{SEP}{Types.HEATING}[1]"
            if heating_col in df[obj_id][Types.HP].columns:
                heating_cop_data.append(df[obj_id][Types.HP][heating_col].values)

                # Only add system ID if we haven't already (to keep lists aligned)
                if len(heating_cop_data) > len(system_ids):
                    system_ids.append(obj_id)

            # Extract DHW COP
            dhw_col = f"{Types.HP}{SEP}{Types.DHW}[1]"
            if dhw_col in df[obj_id][Types.HP].columns:
                dhw_cop_data.append(df[obj_id][Types.HP][dhw_col].values)

    # Create a boxplot for heating COP distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(heating_cop_data, labels=[f"ID {id}" for id in system_ids])
    plt.title("Heating COP Distribution by System")
    plt.ylabel("COP")
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    # Create a boxplot for DHW COP distribution
    plt.subplot(1, 2, 2)
    plt.boxplot(dhw_cop_data, labels=[f"ID {id}" for id in system_ids])
    plt.title("DHW COP Distribution by System")
    plt.ylabel("COP")
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    plt.tight_layout()
    plt.show()

    # Figure 2: Time series visualization for all systems
    # Calculate the number of rows and columns for the subplots
    n_systems = len(df)
    n_cols = min(4, n_systems)
    n_rows = (n_systems + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    # Always flatten the axes array to make it easier to index
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # Make axes iterable if there's only one subplot
    else:
        axes = axes.flatten()  # Flatten the array of axes for easier indexing

    # For each heat pump system, create a separate subplot
    for i, obj_id in enumerate(df):
        if i >= len(axes):
            break  # Safety check

        if Types.HP not in df[obj_id]:
            continue

        # Get system parameters for the title
        config = system_configs.get(obj_id, {})
        hp_source = config.get("hp_source", "Default")
        hp_sink = config.get("hp_sink", "Default")
        temp_sink = config.get("temp_sink", "Default")
        temp_water = config.get("temp_water", "Default")

        # Plot the heating COP time series
        heating_col = f"{Types.HP}{SEP}{Types.HEATING}[1]"
        if heating_col in df[obj_id][Types.HP].columns:
            df[obj_id][Types.HP][heating_col].plot(ax=axes[i], color="#1f77b4", linewidth=1, label="Heating COP")

        # Plot the DHW COP time series
        dhw_col = f"{Types.HP}{SEP}{Types.DHW}[1]"
        if dhw_col in df[obj_id][Types.HP].columns:
            df[obj_id][Types.HP][dhw_col].plot(ax=axes[i], color="#ff7f0e", linewidth=1, label="DHW COP")

        axes[i].set_title(f"ID {obj_id}, Source: {hp_source}, Sink: {hp_sink}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("COP")
        axes[i].set_ylim(0, 12)
        axes[i].legend()
        axes[i].grid(True)

    # Hide empty subplots
    for i in range(len(df), len(axes)):
        if i < len(axes):
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # Figure 3: COP Heatmap (Heating only)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, obj_id in enumerate(df):
        if i >= len(axes):
            break

        if Types.HP not in df[obj_id]:
            continue

        config = system_configs.get(obj_id, {})
        hp_source = config.get("hp_source", "Default")
        hp_sink = config.get("hp_sink", "Default")

        heating_col = f"{Types.HP}{SEP}{Types.HEATING}[1]"
        if heating_col in df[obj_id][Types.HP].columns:
            ts = df[obj_id][Types.HP][heating_col]

            pivot_data = pd.DataFrame({"hour": ts.index.hour, "day_of_year": ts.index.dayofyear, "cop": ts.values})
            pivot_table = pivot_data.pivot_table(values="cop", index="day_of_year", columns="hour", aggfunc="mean")

            im = axes[i].imshow(pivot_table, aspect="auto", cmap="viridis")
            axes[i].set_title(f"ID {obj_id}, Source: {hp_source}, Sink: {hp_sink}")
            axes[i].set_xlabel("Hour of Day")
            axes[i].set_ylabel("Day of Year")

            fig.colorbar(im, ax=axes[i], label="Heating COP")

    for i in range(len(df), len(axes)):
        if i < len(axes):
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # Figure 4: Seasonal daily profile analysis
    seasons = {"Winter": [12, 1, 2], "Spring": [3, 4, 5], "Summer": [6, 7, 8], "Fall": [9, 10, 11]}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, obj_id in enumerate(df):
        if i >= len(axes):
            break

        if Types.HP not in df[obj_id]:
            continue

        config = system_configs.get(obj_id, {})
        hp_source = config.get("hp_source", "Default")
        hp_sink = config.get("hp_sink", "Default")

        heating_col = f"{Types.HP}{SEP}{Types.HEATING}[1]"
        if heating_col in df[obj_id][Types.HP].columns:
            ts_heating = df[obj_id][Types.HP][heating_col]

            for season_name, months in seasons.items():
                season_data = ts_heating[ts_heating.index.month.isin(months)]
                if not season_data.empty:
                    daily_profile = season_data.groupby(season_data.index.hour).mean()
                    axes[i].plot(daily_profile.index, daily_profile.values, label=f"{season_name} (Heating)", linewidth=2)

        dhw_col = f"{Types.HP}{SEP}{Types.DHW}[1]"
        if dhw_col in df[obj_id][Types.HP].columns:
            ts_dhw = df[obj_id][Types.HP][dhw_col]

            for season_name, months in seasons.items():
                season_data = ts_dhw[ts_dhw.index.month.isin(months)]
                if not season_data.empty:
                    daily_profile = season_data.groupby(season_data.index.hour).mean()
                    axes[i].plot(
                        daily_profile.index,
                        daily_profile.values,
                        label=f"{season_name} (DHW)",
                        linewidth=2,
                        linestyle="--",
                    )

        axes[i].set_title(f"ID {obj_id}, Source: {hp_source}, Sink: {hp_sink}")
        axes[i].set_xlabel("Hour of Day")
        axes[i].set_ylabel("Average COP")
        axes[i].set_ylim(0, 10)
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xticks(range(0, 24, 4))

    for i in range(len(df), len(axes)):
        if i < len(axes):
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    # Figure 5: COP vs. Temperature analysis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, obj_id in enumerate(df):
        if i >= len(axes):
            break

        if Types.HP not in df[obj_id]:
            continue

        config = system_configs.get(obj_id, {})
        hp_source = config.get("hp_source", "Default")
        hp_sink = config.get("hp_sink", "Default")

        weather_data = data.get("weather")
        if weather_data is None:
            continue

        if not isinstance(weather_data.index, pd.DatetimeIndex):
            if "datetime" in weather_data.columns:
                weather_data["datetime"] = pd.to_datetime(weather_data["datetime"], utc=True)
                weather_data.set_index("datetime", inplace=True)
            else:
                try:
                    weather_data.index = pd.to_datetime(weather_data.index, utc=True)
                except Exception:
                    datetime_cols = [col for col in weather_data.columns if "time" in col.lower() or "date" in col.lower()]
                    if datetime_cols:
                        weather_data[datetime_cols[0]] = pd.to_datetime(weather_data[datetime_cols[0]], utc=True)
                        weather_data.set_index(datetime_cols[0], inplace=True)
                    else:
                        continue

        temp_col = "air_temperature[C]"

        heating_col = f"{Types.HP}{SEP}{Types.HEATING}[1]"
        if heating_col in df[obj_id][Types.HP].columns:
            merged_data = pd.merge(
                df[obj_id][Types.HP][heating_col],
                weather_data[temp_col],
                left_index=True,
                right_index=True,
                how="inner",
            )
            axes[i].scatter(merged_data[temp_col], merged_data[heating_col], label="Heating COP", alpha=0.5, s=10)

        dhw_col = f"{Types.HP}{SEP}{Types.DHW}[1]"
        if dhw_col in df[obj_id][Types.HP].columns:
            merged_data = pd.merge(
                df[obj_id][Types.HP][dhw_col],
                weather_data[temp_col],
                left_index=True,
                right_index=True,
                how="inner",
            )
            axes[i].scatter(merged_data[temp_col], merged_data[dhw_col], label="DHW COP", alpha=0.5, s=10)

        axes[i].set_title(f"ID {obj_id}, Source: {hp_source}, Sink: {hp_sink}")
        axes[i].set_xlabel("Temperature (°C)")
        axes[i].set_ylabel("COP")
        axes[i].set_ylim(0, 12)
        axes[i].legend()
        axes[i].grid(True)

    for i in range(len(df), len(axes)):
        if i < len(axes):
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    EXAMPLE_DIR = os.path.dirname(__file__)
    objects, data = get_input(EXAMPLE_DIR)

    start = time.perf_counter()
    summary, df = simulate(objects, data, workers=1, path=EXAMPLE_DIR, export=True, plot=True)
    end = time.perf_counter()

    print(f"\n⏱️ Total runtime: {end - start:.4f} s")