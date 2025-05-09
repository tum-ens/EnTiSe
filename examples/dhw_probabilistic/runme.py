"""
Example script demonstrating the probabilistic DHW method and its submethods.

This script demonstrates how to use the probabilistic DHW method and its submethods to generate
domestic hot water demand time series for multiple buildings using different approaches.

Available submethods:
- ProbabilisticDHW: Main facade class that selects the appropriate submethod based on parameters

Available sources:
- jordan_vajen: Based on dwelling size (JordanVajenDwellingSizeDHW)
- hendron_burch: Based on number of occupants (HendronBurchOccupantsDHW)
- iea_annex42: Based on household type (IEAAnnex42HouseholdTypeDHW)
- vdi4655: Includes seasonal cold water temperature variations
- user: Uses user-provided data files with fallbacks

Additional features:
- weekend_activity: Use weekend activity profiles from ASHRAE
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the TimeSeriesGenerator
from entise.core.generator import TimeSeriesGenerator
from entise.constants import Types

# Load data
cwd = '.'  # Current working directory: change if your kernel is not running in the same folder
objects = pd.read_csv(os.path.join(cwd, 'objects.csv'))

# Load weather data from the hvac_rc example
data = {}
weather_file = os.path.join('..', 'hvac_rc', 'data', 'weather.csv')
data['weather'] = pd.read_csv(weather_file, parse_dates=['datetime'])
print('Loaded weather data with shape:', data['weather'].shape)

# Instantiate and configure the generator
gen = TimeSeriesGenerator()
gen.add_objects(objects)

# Generate time series
summary, df = gen.generate(data, workers=1)

# Print summary
print("\nSummary:")
print(summary)

# Create a function to plot DHW demand for a single object
def plot_dhw_demand(ax, obj_id, df, title):
    obj_data = df[obj_id][Types.DHW]
    obj_data.index = pd.to_datetime(obj_data.index)

    # Plot volume
    ax.plot(obj_data.index, obj_data[f'load_{Types.DHW}_volume'], label='DHW Volume (liters)', color='blue')
    ax.set_ylabel('Volume (liters)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True)

    # Add second y-axis for energy
    ax_energy = ax.twinx()
    ax_energy.plot(obj_data.index, obj_data[f'load_{Types.DHW}_energy'], label='DHW Energy (W)', color='red', alpha=0.7)
    ax_energy.set_ylabel('Energy (W)')
    ax_energy.legend(loc='upper right')

    # Format x-axis for better readability
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Show every 7 days

    return obj_data

# Create a 4x2 grid of plots for all 8 objects
fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

# Plot each object
obj1_data = plot_dhw_demand(axes[0], 1, df, 'Building ID: 1 - Jordan & Vajen (Dwelling Size)')
obj2_data = plot_dhw_demand(axes[1], 2, df, 'Building ID: 2 - Hendron & Burch (Occupants)')
obj3_data = plot_dhw_demand(axes[2], 3, df, 'Building ID: 3 - Jordan & Vajen (Dwelling Size)')
obj4_data = plot_dhw_demand(axes[3], 4, df, 'Building ID: 4 - Hendron & Burch (Occupants)')
obj5_data = plot_dhw_demand(axes[4], 5, df, 'Building ID: 5 - IEA Annex 42 (Household Type)')
obj6_data = plot_dhw_demand(axes[5], 6, df, 'Building ID: 6 - Jordan & Vajen with Weekend Activity')
obj7_data = plot_dhw_demand(axes[6], 7, df, 'Building ID: 7 - VDI 4655 (Seasonal Temperature)')
obj8_data = plot_dhw_demand(axes[7], 8, df, 'Building ID: 8 - User-Defined with Fallbacks')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a 4x2 grid for daily profiles
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

# Function to plot daily profile
def plot_daily_profile(ax, obj_data, title):
    obj_data['hour'] = obj_data.index.hour
    daily_profile = obj_data.groupby('hour')[f'load_{Types.DHW}_volume'].mean()
    ax.bar(daily_profile.index, daily_profile.values, color='skyblue', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average DHW Volume (liters)')
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

# Plot daily profiles for each object
plot_daily_profile(axes[0], obj1_data, 'Building ID: 1 - Jordan & Vajen (Dwelling Size)')
plot_daily_profile(axes[1], obj2_data, 'Building ID: 2 - Hendron & Burch (Occupants)')
plot_daily_profile(axes[2], obj3_data, 'Building ID: 3 - Jordan & Vajen (Dwelling Size)')
plot_daily_profile(axes[3], obj4_data, 'Building ID: 4 - Hendron & Burch (Occupants)')
plot_daily_profile(axes[4], obj5_data, 'Building ID: 5 - IEA Annex 42 (Household Type)')
plot_daily_profile(axes[5], obj6_data, 'Building ID: 6 - Jordan & Vajen with Weekend Activity')
plot_daily_profile(axes[6], obj7_data, 'Building ID: 7 - VDI 4655 (Seasonal Temperature)')
plot_daily_profile(axes[7], obj8_data, 'Building ID: 8 - User-Defined with Fallbacks')

plt.tight_layout()
plt.show()

# Compare weekend vs. weekday profiles for object 6 (with weekend activity)
fig, ax = plt.subplots(figsize=(10, 6))

# Extract weekday and weekend data
obj6_data['day_of_week'] = obj6_data.index.dayofweek
weekday_data = obj6_data[obj6_data['day_of_week'] < 5]  # Monday-Friday
weekend_data = obj6_data[obj6_data['day_of_week'] >= 5]  # Saturday-Sunday

# Calculate average profiles
weekday_profile = weekday_data.groupby('hour')[f'load_{Types.DHW}_volume'].mean()
weekend_profile = weekend_data.groupby('hour')[f'load_{Types.DHW}_volume'].mean()

# Plot both profiles
ax.bar(weekday_profile.index - 0.2, weekday_profile.values, width=0.4, color='blue', alpha=0.7, label='Weekday')
ax.bar(weekend_profile.index + 0.2, weekend_profile.values, width=0.4, color='red', alpha=0.7, label='Weekend')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average DHW Volume (liters)')
ax.set_title('Building ID: 6 - Weekday vs. Weekend DHW Profiles')
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print("\nProbabilistic DHW method demonstration completed successfully!")
print("\nAvailable sources:")
print("1. jordan_vajen: Based on dwelling size (JordanVajenDwellingSizeDHW)")
print("2. hendron_burch: Based on number of occupants (HendronBurchOccupantsDHW)")
print("3. iea_annex42: Based on household type (IEAAnnex42HouseholdTypeDHW)")
print("4. vdi4655: Includes seasonal cold water temperature variations")
print("5. user: Uses user-provided data files with fallbacks")
print("6. weekend_activity: Use weekend activity profiles from ASHRAE")

print("\nData sources:")
print("1. Jordan & Vajen (2001): Realistic Domestic Hot-Water Profiles in Different Time Scales")
print("2. Hendron & Burch (2007): Development of Standardized Domestic Hot Water Event Schedules for Residential Buildings (NREL)")
print("3. IEA Annex 42: The Simulation of Building-Integrated Fuel Cell and Other Cogeneration Systems")
print("4. ASHRAE Handbook: HVAC Applications chapter on Service Water Heating")
print("5. VDI 4655: Reference load profiles of single-family and multi-family houses for the use of CHP systems")

print("\nUsage example:")
print("id,dhw,weather,dwelling_size,occupants,household_type,temp_cold,temp_hot,weekend_activity,source")
print("1,ProbabilisticDHW,weather,100,,,10,60,,jordan_vajen")
print("2,ProbabilisticDHW,weather,,4,,10,60,,hendron_burch")
