"""PV generation module based on the pvlib package.

This module implements a photovoltaic (PV) generation method using the pvlib package,
which provides a set of functions and classes for simulating the performance of
photovoltaic energy systems. The implementation follows the Method pattern established
in the project architecture.

The module provides functionality to:
- Process input parameters for PV system configuration
- Validate and prepare weather data
- Calculate PV generation time series based on system parameters and weather data
- Compute summary statistics for the generated time series

The main class, PVLib, inherits from the Method base class and implements the
required interface for integration with the EnTiSe framework.
"""

import logging
import warnings

import pandas as pd
import pvlib
from pvlib import pvsystem

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Types

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Default values for optional keys
POWER = 1  # W
AZIMUTH = 0  # º
TILT = 0  # º
PV_ARRAYS = dict(
        module_parameters=dict(pdc0=1, gamma_pdc=-0.004),
        temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3)
    )
PV_INVERTER = dict(pdc0=3)

class PVLib(Method):
    """Implements a PV generation method based on the pvlib package.

    This class provides functionality to generate photovoltaic (PV) power generation
    time series based on weather data and PV system parameters. It uses the pvlib
    package to model the PV system performance, taking into account factors such as
    solar position, panel orientation, and system efficiency.

    The class follows the Method pattern defined in the EnTiSe framework, implementing
    the required interface for time series generation methods.

    Attributes:
        types (list): List of time series types this method can generate (PV only).
        name (str): Name identifier for the method.
        required_keys (list): Required input parameters (latitude, longitude, weather).
        optional_keys (list): Optional input parameters (power, azimuth, tilt, etc.).
        required_timeseries (list): Required time series inputs (weather).
        optional_timeseries (list): Optional time series inputs (PV arrays).
        output_summary (dict): Mapping of output summary keys to descriptions.
        output_timeseries (dict): Mapping of output time series keys to descriptions.

    Example:
        >>> from entise.methods.pv.pvlib import PVLib
        >>> from entise.core.generator import TimeSeriesGenerator
        >>> 
        >>> # Create a generator and add objects
        >>> gen = TimeSeriesGenerator()
        >>> gen.add_objects(objects_df)  # DataFrame with PV system parameters
        >>> 
        >>> # Generate time series
        >>> summary, timeseries = gen.generate(data)  # data contains weather information
    """
    types = [Types.PV]
    name = "pvlib"
    required_keys = [O.LAT, O.LON, O.WEATHER]
    optional_keys = [O.POWER, O.AZIMUTH, O.TILT, O.ALTITUDE, O.PV_ARRAYS, O.PV_INVERTER]
    required_timeseries = [O.WEATHER]
    optional_timeseries = [O.PV_ARRAYS]
    output_summary = {
            f'{C.GENERATION}_{Types.PV}': 'total PV generation',
            f'{O.GEN_MAX}_{Types.PV}': 'maximum PV generation',
            f'{C.FLH}_{Types.PV}': 'full load hours',
    }
    output_timeseries = {
            f'{C.GENERATION}_{Types.PV}': 'PV generation',
    }

    def generate(self, 
                obj: dict = None, 
                data: dict = None, 
                ts_type: str = Types.PV,
                *, 
                latitude: float = None,
                longitude: float = None,
                weather: pd.DataFrame = None,
                power: float = None,
                azimuth: float = None,
                tilt: float = None,
                altitude: float = None,
                pv_arrays: dict = None,
                pv_inverter: dict = None):
        """Generate PV power time series based on input parameters and weather data.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the PV generation time series,
        and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing PV system parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            ts_type (str, optional): Time series type to generate. Defaults to Types.PV.
            latitude (float, optional): Geographic latitude in degrees. Defaults to None.
            longitude (float, optional): Geographic longitude in degrees. Defaults to None.
            weather (pd.DataFrame, optional): Weather data with solar radiation. Defaults to None.
            power (float, optional): System power rating in watts. Defaults to None.
            azimuth (float, optional): Panel azimuth angle in degrees (0=North, 90=East, 180=South, 270=West). Defaults to None.
            tilt (float, optional): Panel tilt angle in degrees (0=horizontal, 90=vertical). Defaults to None.
            altitude (float, optional): Site altitude in meters. Defaults to None.
            pv_arrays (dict, optional): PV array configuration parameters. Defaults to None.
            pv_inverter (dict, optional): PV inverter configuration parameters. Defaults to None.

        Returns:
            dict: Dictionary containing:
                - "summary" (dict): Summary statistics including total generation,
                  maximum generation, and full load hours.
                - "timeseries" (pd.DataFrame): Time series of PV power generation
                  with timestamps as index.

        Raises:
            Exception: If required data is missing or invalid.

        Example:
            >>> pvlib = PVLib()
            >>> # Using explicit parameters
            >>> result = pvlib.generate(latitude=48.1, longitude=11.6, power=5000, weather=weather_df)
            >>> # Or using dictionaries
            >>> obj = {"latitude": 48.1, "longitude": 11.6, "power": 5000}
            >>> data = {"weather": weather_df}  # DataFrame with solar radiation data
            >>> result = pvlib.generate(obj=obj, data=data)
            >>> summary = result["summary"]
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj, data, 
            latitude=latitude, longitude=longitude, weather=weather,
            power=power, azimuth=azimuth, tilt=tilt, altitude=altitude,
            pv_arrays=pv_arrays, pv_inverter=pv_inverter
        )

        # Continue with existing implementation
        processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)

        logger.debug(f"[PV pvlib]: Generating {ts_type} data")

        timestep = processed_data[O.WEATHER][C.DATETIME].diff().dt.total_seconds().dropna().mode()[0]
        summary = {
            f'{C.GENERATION}_{Types.PV}': (ts.sum() * timestep / 3600).round().astype(int),
            f'{O.GEN_MAX}_{Types.PV}': ts.max().round().astype(int),
            f'{C.FLH}_{Types.PV}': (ts.sum() * timestep / 3600 / processed_obj[O.POWER]).round().astype(int),
        }

        ts = ts.rename(columns={'p_mp': f'{C.POWER}_{Types.PV}'})

        return {
            "summary": summary,
            "timeseries": ts,
        }

def get_input_data(obj, data, method_type=Types.PV):
    """Process and validate input data for PV generation calculation.

    This function extracts required and optional parameters from the input dictionaries,
    applies default values where needed, performs data validation, and prepares the
    data for PV generation calculation.

    Args:
        obj (dict): Dictionary containing PV system parameters such as location,
            orientation, and power rating.
        data (dict): Dictionary containing input data such as weather information.
        method_type (str, optional): Method type to use for prefixing. Defaults to Types.PV.

    Returns:
        tuple: A tuple containing:
            - obj_out (dict): Processed object parameters with defaults applied.
            - data_out (dict): Processed data with required format for calculation.

    Raises:
        Exception: If required weather data is missing.

    Notes:
        - Parameters can be specified with method-specific prefixes (e.g., "pv:altitude")
          which will take precedence over generic parameters (e.g., "altitude").
        - Missing altitude values are automatically looked up based on latitude/longitude.
        - Weather data columns are renamed to match pvlib requirements.
        - Azimuth and tilt values are validated to be within normal ranges.
    """
    obj_out = {
        O.ID: Method.get_with_backup(obj, O.ID),
        O.ALTITUDE: Method.get_with_method_backup(obj, O.ALTITUDE, method_type),
        O.AZIMUTH: Method.get_with_method_backup(obj, O.AZIMUTH, method_type, AZIMUTH),
        O.LAT: Method.get_with_method_backup(obj, O.LAT, method_type),
        O.LON: Method.get_with_method_backup(obj, O.LON, method_type),
        O.POWER: Method.get_with_method_backup(obj, O.POWER, method_type, POWER),
        O.PV_ARRAYS: Method.get_with_method_backup(obj, O.PV_ARRAYS, method_type),
        O.PV_INVERTER: Method.get_with_method_backup(obj, O.PV_INVERTER, method_type),
        O.TILT: Method.get_with_method_backup(obj, O.TILT, method_type, TILT),
    }
    data_out = {
        O.WEATHER: Method.get_with_backup(data, O.WEATHER),
        O.PV_ARRAYS: Method.get_with_backup(data, obj_out[O.PV_ARRAYS], PV_ARRAYS),
        O.PV_INVERTER: Method.get_with_backup(data, O.PV_INVERTER, PV_INVERTER),
    }

    # Fill missing data
    if obj_out[O.ALTITUDE] is None:
        obj_out[O.ALTITUDE] = pvlib.location.lookup_altitude(obj_out[O.LAT], obj_out[O.LON])

    if data_out[O.WEATHER] is not None:
        weather = data_out[O.WEATHER].copy()
        weather.rename(columns={f'{C.SOLAR_GHI}': 'ghi',
                               f'{C.SOLAR_DNI}': 'dni',
                               f'{C.SOLAR_DHI}': 'dhi'},
                       inplace=True, errors='raise')
        weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME], utc=False)
        weather.index = pd.to_datetime(weather[C.DATETIME], utc=True)
        data_out[O.WEATHER] = weather
    else:
        logger.error(f"[PV pvlib]: No weather data")
        raise Exception(f'{O.WEATHER} not available')

    # Sanity checks
    if not 0 <= obj_out[O.AZIMUTH] <= 360:
        logger.warning(f"Azimuth value {obj_out[O.AZIMUTH]} outside normal range [0-360]")
    if not 0 <= obj_out[O.TILT] <= 90:
        logger.warning(f"Tilt value {obj_out[O.TILT]} outside normal range [0-90]")

    return obj_out, data_out

def calculate_timeseries(obj, data):
    """Calculate PV generation time series using the pvlib package.

    This function creates a PV system model based on the input parameters and
    simulates its performance using the provided weather data. It uses the pvlib
    package's ModelChain to perform the simulation.

    Args:
        obj (dict): Dictionary containing processed PV system parameters such as:
            - latitude: Geographic latitude in degrees
            - longitude: Geographic longitude in degrees
            - altitude: Site altitude in meters
            - azimuth: Panel azimuth angle in degrees (0=North, 90=East, 180=South, 270=West)
            - tilt: Panel tilt angle in degrees (0=horizontal, 90=vertical)
            - power: System power rating in watts
        data (dict): Dictionary containing processed data such as:
            - weather: DataFrame with solar radiation data (ghi, dni, dhi)
            - pv_arrays: PV array configuration parameters
            - pv_inverter: PV inverter configuration parameters

    Returns:
        pd.DataFrame: Time series of PV power generation with timestamps as index.

    Notes:
        - The function creates a PV system with a fixed mount at the specified tilt and azimuth.
        - The system is modeled using the pvlib ModelChain with physical AOI model and no spectral losses.
        - The output power is scaled by the system power rating.
    """
    # Get objects
    altitude = obj[O.ALTITUDE]
    azimuth = obj[O.AZIMUTH]
    lat = obj[O.LAT]
    lon = obj[O.LON]
    power = obj[O.POWER]
    tilt = obj[O.TILT]

    # Get data
    df_weather = data[O.WEATHER]
    pv_arrays = data[O.PV_ARRAYS]
    pv_inverter = data[O.PV_INVERTER]

    tz = df_weather[C.DATETIME].iloc[0].tz
    df_weather.index = df_weather.index.tz_convert(tz)
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=altitude, tz=tz)

    # Create the pv system
    array = pvsystem.Array(pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth), **pv_arrays)
    system = pvsystem.PVSystem(arrays=array, inverter_parameters=pv_inverter)
    mc = pvlib.modelchain.ModelChain(system, loc, aoi_model='physical', spectral_model='no_loss')

    mc.run_model(df_weather)

    df = pd.DataFrame(mc.results.ac * power, index=df_weather.index)

    return df
