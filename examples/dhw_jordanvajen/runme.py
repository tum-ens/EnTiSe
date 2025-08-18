"""
Example script demonstrating the DHW method by Jordan et. al. The code is identical to the jupyter notebook.

This script demonstrates how to use the probabilistic DHW method to generate
domestic hot water demand time series for multiple buildings based on dwelling size.

"""

import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the TimeSeriesGenerator
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.generator import TimeSeriesGenerator

# Load data
cwd = "."  # Current working directory: change if your kernel is not running in the same folder
objects = pd.read_csv(os.path.join(cwd, "objects.csv"))
data = {}
common_data_folder = "../common_data"
for file in os.listdir(os.path.join(cwd, common_data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, common_data_folder, file)), parse_dates=True)
print("Loaded data keys:", list(data.keys()))

# Instantiate and configure the generator
gen = TimeSeriesGenerator()
gen.add_objects(objects)

# Generate time series
summary, df = gen.generate(data, workers=1)

# Print summary
print("\nSummary:")
print(summary.to_string())


# Create a function to plot DHW demand for a single object
def plot_dhw_demand(ax, obj_id, df, title):
    obj_data = df[obj_id][Types.DHW]
    obj_data.index = pd.to_datetime(obj_data.index)

    # Plot volume
    ax.plot(obj_data.index, obj_data[f"{Types.DHW}_volume"], label="DHW Volume (liters)", color="blue")
    ax.set_ylabel("Volume (liters)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True)

    # Add second y-axis for energy
    ax_energy = ax.twinx()
    ax_energy.plot(obj_data.index, obj_data[f"{Types.DHW}_energy"], label="DHW Energy (W)", color="red", alpha=0.7)
    ax_energy.set_ylabel("Energy (W)")
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
def plot_daily_profile(ax, obj_data, title):
    obj_data["hour"] = obj_data.index.hour
    daily_profile = obj_data.groupby("hour")[f"{Types.DHW}_volume"].mean()
    ax.bar(daily_profile.index, daily_profile.values, color="skyblue", alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average DHW Volume (liters)")
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    # Calculate yearly total in m3
    yearly_total = obj_data[f"{Types.DHW}_volume"].sum() / 1000  # Convert L to m3
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
plt.show()

# Compare weekend vs. weekday profiles for object 6 (with weekend activity)
fig, ax = plt.subplots(figsize=(10, 6))

# Extract weekday and weekend data
obj6_data = df[6][Types.DHW]
obj6_data["day_of_week"] = obj6_data.index.dayofweek
weekday_data = obj6_data[obj6_data["day_of_week"] < 5]  # Monday-Friday
weekend_data = obj6_data[obj6_data["day_of_week"] >= 5]  # Saturday-Sunday

# Calculate average profiles
weekday_profile = weekday_data.groupby("hour")[f"{Types.DHW}_volume"].mean()
weekend_profile = weekend_data.groupby("hour")[f"{Types.DHW}_volume"].mean()

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
plt.show()
