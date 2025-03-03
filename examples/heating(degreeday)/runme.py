import pandas as pd
from src.core.timeseries_generator import TimeSeriesGenerator

print('This is just an example. The method does not work properly yet.')

# Load objects DataFrame
objects = pd.read_csv('objects.csv')

# Load timeseries dictionary
timeseries_data = {
    "weather": pd.read_csv('weather.csv', parse_dates=True),
}

# Instantiate generator
gen = TimeSeriesGenerator()
gen.add_objects(objects)

# Generate timeseries
summary_df, timeseries_dict = gen.generate_sequential(timeseries_data)

# Display results
print("Summary DataFrame:")
print(summary_df)

print("\nTimeseries Dictionary:")
for obj_id, timeseries in timeseries_dict.items():
    print(f"Object {obj_id} Timeseries:")
    print(timeseries.head())
