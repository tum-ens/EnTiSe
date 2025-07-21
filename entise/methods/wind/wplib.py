"""Wind power generation module based on the windpowerlib package.

This module implements a wind power generation method using the windpowerlib package,
which provides a set of functions and classes for simulating the performance of
wind turbines. The implementation follows the Method pattern established
in the project architecture.

The module provides functionality to:
- Process input parameters for wind turbine configuration
- Validate and prepare weather data
- Calculate wind power generation time series based on system parameters and weather data
- Compute summary statistics for the generated time series

The main class, WindLib, inherits from the Method base class and implements the
required interface for integration with the EnTiSe framework.
"""

import logging

import pandas as pd
from windpowerlib import ModelChain, WindTurbine

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method

logger = logging.getLogger(__name__)

# Default values for optional keys
DEFAULT_TURBINE_TYPE = "SWT130/3600"  # turbine type as in register of windpowerlib
DEFAULT_HUB_HEIGHT = 135  # in m
DEFAULT_POWER = 1  # in W
ROUGHNESS_LENGTH = 0.15

# ModelChain parameters
MODELCHAIN_PARAMS = {
    "wind_speed_model": "logarithmic",  # 'logarithmic' (default),
    # 'hellman' or
    # 'interpolation_extrapolation'
    "density_model": "barometric",  # 'barometric' (default), 'ideal_gas'
    #  or 'interpolation_extrapolation'
    "temperature_model": "linear_gradient",  # 'linear_gradient' (def.) or
    # 'interpolation_extrapolation'
    "power_output_model": "power_curve",  # 'power_curve' (default) or
    # 'power_coefficient_curve'
    "density_correction": False,  # False (default) or True
    "obstacle_height": 0,  # default: 0
    "hellman_exp": None,  # None (default) or None
}


class WPLib(Method):
    """Implements a wind power generation method based on the windpowerlib package.

    This class provides functionality to generate wind power generation time series
    based on weather data and wind turbine parameters. It uses the windpowerlib
    package to model the wind turbine performance, taking into account factors such as
    wind speed, air density, and turbine characteristics.

    The class follows the Method pattern defined in the EnTiSe framework, implementing
    the required interface for time series generation methods.

    Attributes:
        types (list): List of time series types this method can generate (WIND only).
        name (str): Name identifier for the method.
        required_keys (list): Required input parameters (latitude, longitude, weather).
        optional_keys (list): Optional input parameters (power, turbine_type, hub_height, etc.).
        required_timeseries (list): Required time series inputs (weather).
        optional_timeseries (list): Optional time series inputs.
        output_summary (dict): Mapping of output summary keys to descriptions.
        output_timeseries (dict): Mapping of output time series keys to descriptions.

    Example:
        >>> from entise.methods.wind.wplib import WPLib
        >>> from entise.core.generator import TimeSeriesGenerator
        >>>
        >>> # Create a generator and add objects
        >>> gen = TimeSeriesGenerator()
        >>> gen.add_objects(objects_df)  # DataFrame with wind turbine parameters
        >>>
        >>> # Generate time series
        >>> summary, timeseries = gen.generate(data)  # data contains weather information
    """

    types = [Types.WIND]
    name = "wplib"
    required_keys = [O.WEATHER]
    optional_keys = [O.POWER, O.TURBINE_TYPE, O.HUB_HEIGHT, O.WIND_MODEL]
    required_timeseries = [O.WEATHER]
    optional_timeseries = [O.WIND_MODEL]
    output_summary = {
        f"{C.GENERATION}_{Types.WIND}": "total wind power generation",
        f"{O.GEN_MAX}_{Types.WIND}": "maximum wind power generation",
        f"{C.FLH}_{Types.WIND}": "full load hours",
    }
    output_timeseries = {
        f"{C.GENERATION}_{Types.WIND}": "wind power generation",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        ts_type: str = Types.WIND,
        *,
        weather: pd.DataFrame = None,
        power: float = None,
        turbine_type: str = None,
        hub_height: float = None,
    ):
        """Generate wind power time series based on input parameters and weather data.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the wind power generation time series,
        and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing wind turbine parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            ts_type (str, optional): Time series type to generate. Defaults to Types.WIND.
            latitude (float, optional): Geographic latitude in degrees. Defaults to None.
            longitude (float, optional): Geographic longitude in degrees. Defaults to None.
            weather (pd.DataFrame, optional): Weather data with wind speed and direction. Defaults to None.
            power (float, optional): System power rating in watts. Defaults to None.
            turbine_type (str, optional): Type of wind turbine. Defaults to None.
            hub_height (float, optional): Hub height of the turbine in meters. Defaults to None.
            altitude (float, optional): Site altitude in meters. Defaults to None.

        Returns:
            dict: Dictionary containing:
                - "summary" (dict): Summary statistics including total generation,
                  maximum generation, and full load hours.
                - "timeseries" (pd.DataFrame): Time series of wind power generation
                  with timestamps as index.

        Raises:
            Exception: If required data is missing or invalid.

        Example:
            >>> windlib = WPLib()
            >>> # Using explicit parameters
            >>> result = windlib.generate(power=5000, weather=weather_df)
            >>> # Or using dictionaries
            >>> obj = {"power": 5000, "weather": "weather"}
            >>> data = {"weather": weather_df}  # DataFrame with wind data
            >>> result = windlib.generate(obj=obj, data=data)
            >>> summary = result["summary"]
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            weather=weather,
            power=power,
            turbine_type=turbine_type,
            hub_height=hub_height,
        )

        # Continue with existing implementation
        processed_obj, processed_data = get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)

        logger.debug(f"[WIND windlib]: Generating {ts_type} data")

        timestep = processed_data[O.WEATHER].index.diff().total_seconds().dropna()[0]
        summary = {
            f"{C.GENERATION}_{Types.WIND}": (ts.sum() * timestep / 3600).round().astype(int),
            f"{O.GEN_MAX}_{Types.WIND}": ts.max().round().astype(int),
            f"{C.FLH}_{Types.WIND}": (ts.sum() * timestep / 3600 / processed_obj[O.POWER]).round().astype(int),
        }

        ts = ts.rename(columns={O.POWER: f"{C.POWER}_{Types.WIND}"})

        return {
            "summary": summary,
            "timeseries": ts,
        }


def process_weather_data(weather_data):
    """Process weather data to match windpowerlib requirements.

    Args:
        weather_data (pd.DataFrame): Raw weather data.

    Returns:
        pd.DataFrame: Processed weather data with the format required by windpowerlib.
    """
    weather = weather_data.copy()

    # Add roughness length if not present
    if C.ROUGHNESS_LENGTH not in weather.columns:
        weather[C.ROUGHNESS_LENGTH] = ROUGHNESS_LENGTH

    # Select and rename required columns
    weather = weather.loc[
        :, ["temperature_2m", C.SURFACE_PRESSURE, "wind_speed_100m", "wind_direction_100m", C.ROUGHNESS_LENGTH]
    ]
    weather.rename(columns={C.SURFACE_PRESSURE: "pressure"}, inplace=True)

    # Convert temperature to Kelvin
    weather["temperature_2m"] += 273.15

    # Convert surface pressure to Pa
    weather["pressure"] *= 100

    # Create multi-index for columns as needed by windpowerlib
    weather = reshape_weather_data(weather)

    return weather


def reshape_weather_data(df):
    """Reshape the dataframe to have a multi-index for columns as needed by windpowerlib.

    Args:
        df (pd.DataFrame): Weather dataframe to reshape.

    Returns:
        pd.DataFrame: Reshaped dataframe with multi-index columns.
    """
    df.index.name = None

    # Create multi-index for columns
    col_df = df.columns.to_series().str.extract(r"(?P<variable_name>.+?)(?:_(?P<height>\d+)m)?$")
    col_df["height"] = col_df["height"].fillna(0).astype(int)
    df.columns = pd.MultiIndex.from_frame(col_df)

    return df


def get_input_data(obj, data, method_type=Types.WIND):
    """Process and validate input data for wind power generation calculation.

    This function extracts required and optional parameters from the input dictionaries,
    applies default values where needed, performs data validation, and prepares the
    data for wind power generation calculation.

    Args:
        obj (dict): Dictionary containing wind turbine parameters such as location,
            turbine type, and power rating.
        data (dict): Dictionary containing input data such as weather information.
        method_type (str, optional): Method type to use for prefixing. Defaults to Types.WIND.

    Returns:
        tuple: A tuple containing:
            - obj_out (dict): Processed object parameters with defaults applied.
            - data_out (dict): Processed data with required format for calculation.

    Raises:
        Exception: If required weather data is missing.

    Notes:
        - Parameters can be specified with method-specific prefixes (e.g., "wind:altitude")
          which will take precedence over generic parameters (e.g., "altitude").
        - Weather data is processed to match the format required by windpowerlib.
    """
    obj_out = {
        O.ID: Method.get_with_backup(obj, O.ID),
        O.POWER: Method.get_with_method_backup(obj, O.POWER, method_type, DEFAULT_POWER),
        O.TURBINE_TYPE: Method.get_with_method_backup(obj, O.TURBINE_TYPE, method_type, DEFAULT_TURBINE_TYPE),
        O.HUB_HEIGHT: Method.get_with_method_backup(obj, O.HUB_HEIGHT, method_type, DEFAULT_HUB_HEIGHT),
    }

    data_out = {
        O.WEATHER: Method.get_with_backup(data, O.WEATHER),
    }

    # Process weather data
    if data_out[O.WEATHER] is not None:
        weather = data_out[O.WEATHER].copy()
        weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME], utc=False)
        weather.index = pd.to_datetime(weather[C.DATETIME], utc=True)
        data_out[O.WEATHER] = process_weather_data(weather)
    else:
        logger.error("[WIND windlib]: No weather data")
        raise Exception(f"{O.WEATHER} not available")

    return obj_out, data_out


def calculate_timeseries(obj, data):
    """Calculate wind power generation time series using the windpowerlib package.

    This function creates a wind turbine model based on the input parameters and
    simulates its performance using the provided weather data. It uses the windpowerlib
    package's ModelChain to perform the simulation.

    Args:
        obj (dict): Dictionary containing processed wind turbine parameters such as:
            - latitude: Geographic latitude in degrees
            - longitude: Geographic longitude in degrees
            - turbine_type: Type of wind turbine
            - hub_height: Hub height of the turbine in meters
            - power: System power rating in watts
        data (dict): Dictionary containing processed data such as:
            - weather: DataFrame with wind speed and direction data

    Returns:
        pd.DataFrame: Time series of wind power generation with timestamps as index.

    Notes:
        - The function creates a wind turbine with the specified type and hub height.
        - The system is modeled using the windpowerlib ModelChain with the parameters
          defined in MODELCHAIN_PARAMS.
        - The output power is scaled by the system power rating.
    """
    # Get objects
    power = obj[O.POWER] / 1e3
    turbine_type = obj[O.TURBINE_TYPE]
    hub_height = obj[O.HUB_HEIGHT]

    # Get data
    weather = data[O.WEATHER]

    # Create turbine model
    turbine = WindTurbine(
        turbine_type=turbine_type,
        hub_height=hub_height,
    )

    # Create model chain and run model
    mc = ModelChain(turbine, **MODELCHAIN_PARAMS).run_model(weather)

    # Get output power
    power_output = mc.power_output

    # Set time index to origin timestamp
    power_output.index.name = C.DATETIME
    power_output.name = O.POWER

    # Create DataFrame and rename column
    df = pd.DataFrame(power_output, columns=[O.POWER])

    # Calculate and round power (normalize by nominal power)
    df[O.POWER] = df[O.POWER] / turbine.nominal_power * power * 1e3
    df = df.round(3)

    return df
