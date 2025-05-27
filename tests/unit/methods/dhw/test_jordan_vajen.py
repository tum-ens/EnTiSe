"""
Test script for the JordanVajen class.

This script tests the JordanVajen class by generating DHW demand time series
for a sample dwelling and printing the results.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from entise.methods.dhw.jordan_vajen import JordanVajen
from entise.constants import Columns as C, Objects as O, Types

# Create sample weather data
def create_sample_weather(start_date='2023-01-01', days=7):
    """Create sample weather data for testing."""
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=days)
    index = pd.date_range(start=start, end=end, freq='H')

    # Create a DataFrame with temperature data
    weather = pd.DataFrame({
        C.DATETIME: index,
        C.TEMP_OUT: np.random.normal(10, 5, len(index))  # Random outdoor temperatures
    }, index=index)

    return weather

# Create sample object parameters
def create_sample_obj():
    """Create sample object parameters for testing."""
    return {
        O.ID: 'test_obj',
        O.DWELLING_SIZE: 100,  # 100 m²
        O.TEMP_WATER_COLD: 10,       # 10°C
        O.TEMP_WATER_HOT: 60,        # 60°C
        O.DATETIMES: 'weather',
        O.SEED: 42  # Add a seed for reproducibility
    }

# Main test function
def test_jordan_vajen_dhw():
    """Test the JordanVajen class."""
    print("Testing JordanVajen class...")

    # Create sample data
    weather = create_sample_weather()
    obj = create_sample_obj()
    data = {O.WEATHER: weather}

    # Create JordanVajen instance
    dhw = JordanVajen()

    # Generate DHW demand time series
    result = dhw.generate(obj, data)

    # Print summary
    print("\nSummary:")
    for key, value in result['summary'].items():
        print(f"  {key}: {value}")

    # Print first few rows of time series
    print("\nTime series (first 24 hours):")
    print(result['timeseries'].head(24))

    # Plot time series if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Plot volume
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(result['timeseries'].index, result['timeseries'][f'{C.LOAD}_{Types.DHW}_volume'])
        plt.title('DHW Volume Demand')
        plt.ylabel('Liters')
        plt.grid(True)

        # Plot energy
        plt.subplot(2, 1, 2)
        plt.plot(result['timeseries'].index, result['timeseries'][f'{C.LOAD}_{Types.DHW}_energy'])
        plt.title('DHW Energy Demand')
        plt.ylabel('Watts')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not available for plotting.")

    return result

if __name__ == "__main__":
    test_jordan_vajen_dhw()
