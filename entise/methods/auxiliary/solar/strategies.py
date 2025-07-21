import numpy as np
import pandas as pd
import pvlib

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base_auxiliary import AuxiliaryMethod


class SolarGainsInactive(AuxiliaryMethod):
    """
    Handles the creation of solar gains data for cases where solar gains are inactive.

    This class is used to define an auxiliary method that generates a time-series
    of solar gains data with zero values when solar gains calculations are not
    applicable. It ensures compatibility with the required input data and produces
    an output in a specific format.
    """

    required_timeseries = [O.WEATHER]

    def get_input_data(self, obj, data):
        """
        Processes input data and extracts specified information for further usage.

        Args:
            obj: A reference to an object that may hold contextual information or be used
                in processing, nature of its usage to be defined by implementation.
            data: Dictionary or mapping that contains information from which specific
                values, such as weather data, are extracted.

        Returns:
            Dictionary containing the extracted weather data under a predefined key.
        """
        return {O.WEATHER: data[O.WEATHER]}

    def run(self, weather):
        """
        Processes weather data to compute a DataFrame with solar gains.

        This function calculates a DataFrame containing a single column of zeros that
        represents the solar gains. The DataFrame uses the same index as the provided
        weather data. This is primarily used in scenarios where solar gains need to
        be initialized or simulated with a default value.

        Args:
            weather: A pandas DataFrame representing weather data. The index should
                be a time-based index, and the length of the DataFrame determines the
                number of rows in the resultant DataFrame.

        Returns:
            pandas.DataFrame: A DataFrame with a single column named 'O.GAINS_SOLAR',
            filled with zeros of type `np.float32`. The index corresponds to the input
            weather DataFrame's index.
        """
        return pd.DataFrame({O.GAINS_SOLAR: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)


class SolarGainsPVLib(AuxiliaryMethod):
    """
    Perform calculations of solar gains for buildings using irradiance models.

    This class provides methods to process input data and calculate solar gains by considering
    weather conditions, window configurations, and solar irradiance models. It integrates
    with `pvlib` to compute solar positions and irradiance values. The class supports different
    irradiance models, such as "isotropic" and "haydavies", and handles missing input gracefully.
    """

    required_keys = [O.ID, O.LAT, O.LON]
    optional_keys = ["model"]
    required_timeseries = [O.WEATHER, O.WINDOWS]

    def get_input_data(self, obj, data):
        object_id = obj[O.ID]
        windows = data.get(O.WINDOWS, None)
        if windows is not None:
            windows = windows.loc[windows[O.ID] == object_id]
            windows = windows if not windows.empty else None
        input_data = {
            O.LAT: obj[O.LAT],
            O.LON: obj[O.LON],
            "model": obj.get("model", "isotropic"),
            O.WEATHER: data[O.WEATHER],
            O.WINDOWS: windows,
        }
        return input_data

    def run(self, weather, windows, latitude, longitude, model="isotropic"):
        """Calculate solar gains for a building.

        Args:
            weather (pd.DataFrame): Weather data.
            windows (pd.DataFrame): Windows data.
            latitude (float): Latitude.
            longitude (float): Longitude.
            model (str, optional): Irradiance model to use. Default is "isotropic".

        Returns:
            pd.DataFrame: Solar gains for each timestep.

        Raises:
            ValueError: If the irradiance model is unknown.
        """
        if windows is None:
            return pd.DataFrame({O.GAINS_SOLAR: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)

        # Obtain all relevant information upfront
        timezone = weather.index[0].tzinfo or "UTC"
        location = pvlib.location.Location(latitude, longitude, tz=timezone)
        solpos = location.get_solarposition(pd.to_datetime(weather.index, utc=True), method="nrel_numba")

        # Calculate values depending on model
        if model == "haydavies":
            dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
            dni = pvlib.irradiance.dirint(
                ghi=weather[C.SOLAR_GHI], solar_zenith=solpos["apparent_zenith"], times=weather.index
            ).fillna(0)
        elif model == "isotropic":
            dni_extra = None
            dni = weather[C.SOLAR_DNI]
        else:
            raise ValueError("Unknown irradiance model.")

        total_solar_gains = np.zeros(len(weather), dtype=np.float32)
        for _, window in windows.iterrows():
            # Compute irradiance for this window
            irr = pvlib.irradiance.get_total_irradiance(
                surface_tilt=window[C.TILT],
                surface_azimuth=window[C.ORIENTATION],
                solar_zenith=solpos["zenith"],
                solar_azimuth=solpos["azimuth"],
                dni=dni,
                ghi=weather[C.SOLAR_GHI],
                dhi=weather[C.SOLAR_DHI],
                dni_extra=dni_extra,
                model=model,
            )
            poa_global = irr["poa_global"]
            window_gains = poa_global * window["area"] * window["transmittance"] * window["shading"]

            # Accumulate the gains
            total_solar_gains += window_gains.to_numpy(dtype=np.float32)
        return pd.DataFrame({O.GAINS_SOLAR: total_solar_gains}, index=weather.index)


__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, type)
    and issubclass(obj, AuxiliaryMethod)
    and obj is not AuxiliaryMethod  # exclude the base class
]
