"""Creates pu time series data for the PV generation for a given year and location"""

import pandas as pd
import os
import pvlib
from time import perf_counter
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from pvlib import pvsystem


def main(df_weather: pd.DataFrame, lat: float, lon: float, location: str, year: int, azimuth: list, tilt: list) \
        -> pd.DataFrame:
    """Creates pu time series data for the PV generation for a given year and location"""
    # Create the location object
    altitude = pvlib.location.lookup_altitude(lat, lon)
    tz = df_weather.index[0].tz
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=altitude, name=location, tz=tz)

    # Create the angle combinations
    angles = list(itertools.product(azimuth, tilt))

    # Create the pv system
    array_kwargs = dict(
        module_parameters=dict(pdc0=1, gamma_pdc=-0.004),
        temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3)
    )

    results = dict()

    # Loop through the angles
    for angle in tqdm(angles):
        # Create the pv system
        array = pvsystem.Array(pvsystem.FixedMount(surface_tilt=angle[1], surface_azimuth=angle[0]), **array_kwargs)
        system = pvsystem.PVSystem(arrays=array, inverter_parameters=dict(pdc0=3),
                                   name=f'azimuth_{angle[0]}_tilt_{angle[1]}')
        mc = pvlib.modelchain.ModelChain(system, loc, aoi_model='physical',
                                   spectral_model='no_loss')

        # Get clear sky data
        # clearsky = loc.get_clearsky(df_weather.index)

        # Run the model
        mc.run_model(df_weather)

        # Add the results to the dataframe
        results[f'azimuth_{angle[0]}_tilt_{angle[1]}'] = mc.results.ac

    df_pv = pd.DataFrame(results, index=df_weather.index)

    return df_pv


if __name__ == '__main__':
    latitude = 49.73637771606445  # Forchheim latitude
    longitude = 11.074522972106934  # Forchheim longitude
    location = 'Forchheim'
    year = 2019
    azimuth = list(range(0, 360 + 1, 45))
    tilt = list(range(0, 90 + 1, 15))

    # Doing it with the open-meteo data format
    df_weather = pd.read_csv('../weather/weather.csv', index_col=0, parse_dates=True)
    df_weather.index = pd.to_datetime(df_weather.index, utc=True).tz_convert('Europe/Berlin')

    instant = False
    if instant:
        str_instant = '_instant'
    else:
        str_instant = ''

    df_weather.rename(columns={f'shortwave_radiation{str_instant}': 'ghi',
                               f'direct_normal_irradiance{str_instant}': 'dni',
                               f'diffuse_radiation{str_instant}': 'dhi'}, inplace=True)

    print(df_weather.head().to_string())

    # # Has to be shifted by one hour back due to pvlibs strange integration
    # df_weather.index = df_weather.index - pd.Timedelta(hours=1)

    # Calculate the PV generation
    df_pv = main(df_weather, latitude, longitude, location, year, azimuth, tilt)
    print('ALWAYS CHECK FOR CONSISTENT PATTERNS')

    # Save the results
    folder = 'file'
    df_pv.to_csv(f'{folder}/pv_pu_{location}_{year}.csv')
    split = True
    if split:
        for col in df_pv.columns:
            df_col = df_pv[[col]]
            df_col.rename(columns={col: 'power'}, inplace=True)
            df_col.insert(0, 'timestamp', df_col.index.astype('int64') // 10**9)
            df_col.to_csv(f'{folder}/{col}.csv', index=True)
    style = 'Y'
    df_pv = df_pv.resample(style).sum()
    if style == 'Y':
        df_pv.plot(kind='bar', figsize=(12, 6))
    else:
        df_pv.plot(kind='line', figsize=(12, 6))
    # df_pv.iloc[180*24:187*24].plot(kind='line', figsize=(12, 6))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    # Calculate hourly mean values of ghi, dni, dhi
    df_weather = df_weather.groupby(df_weather.index.hour).mean()
    df_weather['ghi'].plot(kind='line', figsize=(12, 6))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(ticks=range(0, 24 + 1, 2))
    plt.xlim(0, 24)
    plt.grid()
    plt.tight_layout()
    plt.show()

    # for col in df_pv.columns:
    #     if 'azimuth' in col:
    #         print(f'{col}: {df_pv[col].sum().astype(int)}')
