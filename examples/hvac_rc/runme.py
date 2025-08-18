"""
Example script demonstrating the pipeline architecture.

This script demonstrates how to use the pipeline architecture to generate
time series for multiple buildings.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from entise.constants import Columns as Cols

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the new TimeSeriesGenerator
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
data_folder = "data"
for file in os.listdir(os.path.join(cwd, data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, data_folder, file)), parse_dates=True)
print("Loaded data keys:", list(data.keys()))

# Instantiate and configure the generator
gen = TimeSeriesGenerator()

# Define the building ID to process and visualize
building_id = 31991690  # Select the object you want to simulate

gen.add_objects(objects)

# Generate time series
summary, df = gen.generate(data, workers=1)

# Print summary
print("Summary:")
summary_kwh = (summary / 1000).round(0).astype(int)
summary_kwh.rename(columns=lambda x: x.replace("[W]", "[kW]").replace("[Wh]", "[kWh]"), inplace=True)
print(summary_kwh.to_string())

# Visualize results for the processed building
# Note: We're using the same building_id as defined above
building_data = df[building_id][Types.HVAC]

# Figure 1: Indoor & Outdoor Temperature and Solar Radiation (GHI)
fig, ax1 = plt.subplots(figsize=(15, 6))

# Solar radiation plot (GHI) with separate y-axis
ax2 = ax1.twinx()
ax2.plot(
    building_data.index, data["weather"][Cols.SOLAR_GHI],
    label="Solar Radiation (GHI)", color="tab:orange", alpha=0.3
)
ax2.set_ylabel("Solar Radiation (W/m²)")
ax2.legend(loc="upper right")
ax2.set_ylim(-250, 1000)

# Temperature plot
ax1.plot(building_data.index, data["weather"][f"{Cols.TEMP_AIR}@2m"], label="Outdoor Temp", color="tab:cyan", alpha=0.7)
ax1.plot(building_data.index, building_data[Cols.TEMP_IN], label="Indoor Temp", color="tab:blue")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title(f"Building ID: {building_id} - Temperatures and Solar Radiation")
ax1.legend(loc="upper left")
ax1.grid(True)
ax1.set_ylim(-10, 40)

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)  # required to see through ax1 to ax2
plt.tight_layout()
plt.show()

# Figure 2: Heating and Cooling Loads
fig, ax = plt.subplots(figsize=(14, 5))
heating_MWh = summary.loc[building_id, "demand_heating[Wh]"] / 1e6
cooling_MWh = summary.loc[building_id, "demand_cooling[Wh]"] / 1e6
(line1,) = ax.plot(
    building_data.index,
    building_data["load_heating[W]"],
    label=f"Heating: {heating_MWh:.1f} MWh",
    color="tab:red",
    alpha=0.8,
)
(line2,) = ax.plot(
    building_data.index,
    building_data["load_cooling[W]"],
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
plt.show()

# Figure 3: Outdoor Temperature with Heating & Cooling Loads
fig, ax1 = plt.subplots(figsize=(15, 6))

# Plot outdoor temperature on left y-axis
air_temp = data["weather"][f"{Cols.TEMP_AIR}@2m"]
ax1.plot(building_data.index, air_temp
         , label="Outdoor Temp", color="tab:cyan", alpha=0.7)

ax1.set_ylabel("Outdoor Temp (°C)")
ax1.set_ylim(air_temp.min().round() - 2, air_temp.max().round() + 2)

# Create second y-axis for loads
ax2 = ax1.twinx()
ax2.plot(building_data.index, building_data["load_heating[W]"], label="Heating Load", color="tab:red", alpha=0.8)
ax2.plot(building_data.index, building_data["load_cooling[W]"], label="Cooling Load", color="tab:blue", alpha=0.8)
ax2.set_ylabel("HVAC Load (W)")
ax2.set_ylim(
    min(building_data["load_heating[W]"].min(), building_data["load_cooling[W]"].min()) * 1.1,
    max(building_data["load_heating[W]"].max(), building_data["load_cooling[W]"].max()) * 1.1,
)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title(f"Building ID: {building_id} - Outdoor Temp & HVAC Loads")
ax1.grid(True)
fig.tight_layout()
plt.show()
