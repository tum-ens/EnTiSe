"""Heat pump COP calculation based on Ruhnau et al. (2019).

This module implements a heat pump COP (Coefficient of Performance) calculation method
based on the paper by Ruhnau et al. (2019) "Time series of heat demand and heat pump
efficiency for energy system modeling".

The implementation follows the Method pattern established in the project architecture
and provides functionality to calculate COP time series for different heat pump types
(air, ground, water source) and heating systems (floor heating, radiator, water heating)
based on temperature differences.
"""

import logging

import pandas as pd

import entise.methods.hp.defaults as defs
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base import Method

logger = logging.getLogger(__name__)

# Define default heat pump parameters for each technology
HP_SYSTEM = {
    defs.HP_AIR: {
        "a": 0.0005,
        "b": -0.09,
        "c": 6.08,
    },
    defs.HP_SOIL: {
        "a": 0.0012,
        "b": -0.21,
        "c": 10.29,
    },
    defs.HP_WATER: {
        "a": 0.0012,
        "b": -0.20,
        "c": 9.97,
    },
}

# Define default temperature parameters for each sink type
DEFAULT_SINKS = {
    defs.FLOOR: {O.TEMP_SINK: 30, O.GRADIENT_SINK: -0.5},
    defs.RADIATOR: {O.TEMP_SINK: 40, O.GRADIENT_SINK: -1.0},
    defs.WATER: {O.TEMP_SINK: 50, O.GRADIENT_SINK: 0},
}

# Default correction factor
DEFAULT_CORRECTION_FACTOR = 0.85

# Strings
HP_PARAMS = "heat_pump_parameters"
HEAT_PARAMS = "heating_parameters"
DHW_PARAMS = "dhw_parameters"


class Ruhnau(Method):
    """Heat pump COP calculation based on Ruhnau et al. (2019).

    Implements COP calculations for different heat pump types (air, ground, water source)
    and heating systems (floor heating, radiator, water heating) based on temperature differences.

    Attributes:
        types (list): List of time series types this method can generate (HP only).
        name (str): Name identifier for the method.
        required_keys (list): Required input parameters (weather).
        optional_keys (list): Optional input parameters (hp_source, hp_sink, hp_temp, correction_factor).
        required_timeseries (list): Required time series inputs (weather).
        optional_timeseries (list): Optional time series inputs (none).
        output_summary (dict): Mapping of output summary keys to descriptions.
        output_timeseries (dict): Mapping of output time series keys to descriptions.
    """

    types = [Types.HP]
    name = "ruhnau"
    required_keys = [O.WEATHER]
    optional_keys = [
        O.HP_SOURCE,
        O.HP_SINK,
        O.TEMP_SINK,
        O.GRADIENT_SINK,
        O.TEMP_WATER,
        O.CORRECTION_FACTOR,
        O.HP_SYSTEM,
    ]
    required_timeseries = [O.WEATHER]
    optional_timeseries = [O.HP_SYSTEM]
    output_summary = {
        f"{Types.HP}{SEP}{Types.HEATING}_avg[1]": "average heating COP value",
        f"{Types.HP}{SEP}{Types.HEATING}_min[1]": "minimum heating COP value",
        f"{Types.HP}{SEP}{Types.HEATING}_max[1]": "maximum heating COP value",
        f"{Types.HP}{SEP}{Types.DHW}_avg[1]": "average DHW COP value",
        f"{Types.HP}{SEP}{Types.DHW}_min[1]": "minimum DHW COP value",
        f"{Types.HP}{SEP}{Types.DHW}_max[1]": "maximum DHW COP value",
    }
    output_timeseries = {
        f"{Types.HP}{SEP}{Types.HEATING}[1]": "heating COP time series",
        f"{Types.HP}{SEP}{Types.DHW}[1]": "DHW COP time series",
    }

    def generate(
        self,
        obj: dict = None,
        data: dict = None,
        ts_type: str = Types.HP,
        *,
        weather: pd.DataFrame = None,
        hp_source: str = None,
        hp_sink: str = None,
        temp_sink: float = None,
        gradient_sink: float = None,
        temp_water: float = None,
        correction_factor: float = None,
        hp_system: dict = None,
        cop_coefficients: dict = None,
    ):
        """Generate heat pump COP time series for both heating and DHW.

        Args:
            obj (dict, optional): Dictionary with heat pump parameters
            data (dict, optional): Dictionary with input data
            ts_type (str, optional): Time series type to generate
            weather (pd.DataFrame, optional): Weather data with temperatures
            hp_source (str, optional): Heat pump source type ('ASHP', 'GSHP', 'WSHP')
            hp_sink (str, optional): Heat sink type ('floor', 'radiator')
            temp_sink (float, optional): Temperature setting for heating
            gradient_sink (float, optional): Gradient setting for heating
            temp_water (float, optional): Temperature setting for DHW
            correction_factor (float, optional): Efficiency correction factor
            hp_system (dict, optional): System configuration for heat pump
            cop_coefficients (dict, optional): Custom coefficients for COP calculation (a, b, c)

        Returns:
            dict: Dictionary with summary statistics and COP time series for both heating and DHW

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Process inputs
        processed_obj, processed_data = self._process_kwargs(
            obj,
            data,
            weather=weather,
            hp_source=hp_source,
            hp_sink=hp_sink,
            hp_temp=temp_sink,
            gradient_sink=gradient_sink,
            temp_water=temp_water,
            correction_factor=correction_factor,
            cop_coefficients=cop_coefficients,
        )

        # Get input data with validation
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data)

        # Calculate heating COP time series
        heating_cop_series = _calculate_heating_cop_series(processed_obj, processed_data)

        # Calculate DHW COP time series
        dhw_cop_series = _calculate_dhw_cop_series(processed_obj, processed_data)

        return self._format_output(heating_cop_series, dhw_cop_series, processed_data)

    def _get_input_data(self, obj, data):
        """Process and validate input parameters.

        Args:
            obj (dict): Dictionary with heat pump parameters
            data (dict): Dictionary with input data

        Returns:
            tuple: Processed object, data, and custom coefficients

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        obj_out = {}
        obj_out[HP_PARAMS] = {}
        obj_out[HEAT_PARAMS] = {}
        obj_out[DHW_PARAMS] = {}
        data_out = {}

        # Get system parameters either from obj df or specifications
        hp_system = self.get_with_backup(obj, O.HP_SYSTEM, None)
        if isinstance(hp_system, str):
            params = self.get_with_backup(data, hp_system)
            obj_out[HP_PARAMS] = params[HP_PARAMS]
            obj_out[HEAT_PARAMS] = params[HEAT_PARAMS]
            obj_out[DHW_PARAMS] = params[DHW_PARAMS]
        else:
            hp_source = self.get_with_backup(obj, O.HP_SOURCE, defs.DEFAULT_HP)
            obj_out[HP_PARAMS][O.HP_SOURCE] = hp_source

            # Check for custom coefficients first
            cop_coefficients = self.get_with_backup(obj, "cop_coefficients", None)
            obj_out[HP_PARAMS]["coeffs"] = cop_coefficients if cop_coefficients is not None else HP_SYSTEM[hp_source]
            hp_sink = self.get_with_backup(obj, O.HP_SINK, defs.DEFAULT_HEAT)
            obj_out[HEAT_PARAMS][O.HP_SINK] = hp_sink
            obj_out[HEAT_PARAMS][O.TEMP_SINK] = self.get_with_backup(
                obj, O.TEMP_SINK, DEFAULT_SINKS[hp_sink][O.TEMP_SINK]
            )
            obj_out[HEAT_PARAMS][O.GRADIENT_SINK] = self.get_with_backup(
                obj, O.GRADIENT_SINK, DEFAULT_SINKS[hp_sink][O.GRADIENT_SINK]
            )
            dhw_sink = defs.WATER
            obj_out[DHW_PARAMS][O.HP_SINK] = dhw_sink
            obj_out[DHW_PARAMS][O.TEMP_SINK] = self.get_with_backup(
                obj, O.TEMP_WATER, DEFAULT_SINKS[dhw_sink][O.TEMP_SINK]
            )
            obj_out[DHW_PARAMS][O.GRADIENT_SINK] = DEFAULT_SINKS[dhw_sink][O.GRADIENT_SINK]

        obj_out[O.CORRECTION_FACTOR] = self.get_with_backup(obj, O.CORRECTION_FACTOR, DEFAULT_CORRECTION_FACTOR)

        # Process weather data
        weather = self.get_with_backup(obj, O.WEATHER, None)
        if weather is None:
            logger.warning("No weather provided")
            raise ValueError("Weather data is required but not provided")
        weather = self.get_with_backup(data, weather)
        weather = self._strip_weather_height(weather)
        data_out[O.WEATHER] = self._process_weather_data(weather)

        return obj_out, data_out

    def _process_weather_data(self, df):
        """Prepare weather data for COP calculation.

        Args:
            df (pd.DataFrame): Weather data

        Returns:
            pd.DataFrame: Processed weather data with required temperature columns
        """
        # Ensure datetime index
        if C.DATETIME in df.columns:
            df.index = pd.to_datetime(df[C.DATETIME])
        else:
            df.index = pd.to_datetime(df.index)

        # Add missing temperature columns
        df = self._add_missing_temperature_columns(df)

        return df

    @staticmethod
    def _add_missing_temperature_columns(df):
        """Add missing temperature columns to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to process

        Returns:
            pd.DataFrame: DataFrame with missing temperature columns added
        """
        # Check if the air temperature column is missing
        if C.TEMP_AIR not in df.columns:
            logger.warning("No air temperature column found in weather data")
            raise Warning(f"No air temperature column '{C.TEMP_AIR}' found in weather data")

        # Add soil temperature column if missing
        if C.TEMP_SOIL not in df.columns:
            logger.info("Calculating soil temperature from air temperature")
            df["rolling_temp"] = df[C.TEMP_AIR].rolling(window=24, min_periods=1).mean()
            df[C.TEMP_SOIL] = df["rolling_temp"].apply(_calc_soil_temp)
            df.drop(columns=["rolling_temp"], inplace=True)

        # Add groundwater temperature column if missing
        if C.TEMP_WATER_GROUND not in df.columns:
            logger.info("Setting default groundwater temperature to 10°C")
            df[C.TEMP_WATER_GROUND] = 10

        return df

    @staticmethod
    def _format_output(heating_cop_series, dhw_cop_series, processed_data):
        """Format the output dictionary for the method.

        Args:
            summary (dict): Summary statistics
            timeseries (pd.DataFrame): Time series data

        Returns:
            dict: Formatted output dictionary
        """

        # Prepare output
        summary = {
            f"{Types.HP}{SEP}{Types.HEATING}_avg[1]": heating_cop_series.mean().round(2),
            f"{Types.HP}{SEP}{Types.HEATING}_min[1]": heating_cop_series.min().round(2),
            f"{Types.HP}{SEP}{Types.HEATING}_max[1]": heating_cop_series.max().round(2),
            f"{Types.HP}{SEP}{Types.DHW}_avg[1]": dhw_cop_series.mean().round(2),
            f"{Types.HP}{SEP}{Types.DHW}_min[1]": dhw_cop_series.min().round(2),
            f"{Types.HP}{SEP}{Types.DHW}_max[1]": dhw_cop_series.max().round(2),
        }

        timeseries = pd.DataFrame(
            {
                f"{Types.HP}{SEP}{Types.HEATING}[1]": heating_cop_series,
                f"{Types.HP}{SEP}{Types.DHW}[1]": dhw_cop_series,
            },
            index=processed_data[O.WEATHER].index,
        )

        return {
            "summary": summary,
            "timeseries": timeseries,
        }


def _calc_soil_temp(t_avg):
    """Calculate soil temperature based on average air temperature.

    Based on "WP Monitor" Feldmessung von Wärmepumpenanlagen, Frauenhofer ISE, 2014.

    Args:
        t_avg (float): Average air temperature

    Returns:
        float: Calculated soil temperature
    """
    t_soil = -0.0003 * t_avg**3 + 0.0086 * t_avg**2 + 0.3047 * t_avg + 5.0647
    return t_soil


def _calculate_heating_cop_series(obj, data):
    """Calculate heating COP time series based on temperature differences.

    Args:
        obj (dict): Dictionary with heat pump parameters
        data (dict): Dictionary with input data

    Returns:
        pd.Series: Heating COP time series
    """
    weather_df = data[O.WEATHER]
    hp_source = obj[HP_PARAMS][O.HP_SOURCE]
    hp_coeffs = obj[HP_PARAMS]["coeffs"]
    temp_sink = obj[HEAT_PARAMS][O.TEMP_SINK]
    gradient_sink = obj[HEAT_PARAMS][O.GRADIENT_SINK]
    correction_factor = obj[O.CORRECTION_FACTOR]
    return _calculate_cop(weather_df, hp_source, hp_coeffs, temp_sink, gradient_sink, correction_factor)


def _calculate_dhw_cop_series(obj, data):
    """Calculate DHW COP time series based on temperature differences.

    Args:
        obj (dict): Dictionary with heat pump parameters
        data (dict): Dictionary with input data

    Returns:
        pd.Series: DHW COP time series
    """
    weather_df = data[O.WEATHER]
    hp_source = obj[HP_PARAMS][O.HP_SOURCE]
    hp_coeffs = obj[HP_PARAMS]["coeffs"]
    temp_sink = obj[DHW_PARAMS][O.TEMP_SINK]
    gradient_sink = obj[DHW_PARAMS][O.GRADIENT_SINK]
    correction_factor = obj[O.CORRECTION_FACTOR]
    return _calculate_cop(weather_df, hp_source, hp_coeffs, temp_sink, gradient_sink, correction_factor)


def _calculate_cop(
    weather_df,
    hp_source,
    coeffs,
    temp_sink,
    gradient_sink,
    correction_factor,
    temp_offset_types: list = (defs.HP_SOIL, defs.HP_WATER),
    temp_offset: float = 5,
):
    """Calculate COP time series based on temperature differences.

    Args:
        weather_df (pd.DataFrame): Weather data containing temperature information
        hp_source (str): Heat pump source type
        coeffs (dict): Coefficients for the COP formula
        temp_sink (float): Temperature setting for the heat sink
        gradient_sink (float): Gradient for sink temperature calculation
        correction_factor (float): Efficiency correction factor
        temp_offset_types (list, optional): Heat pump source types that need temperature offset
        temp_offset (float, optional): Temperature offset value

    Returns:
        pd.Series: COP time series
    """
    # Calculate sink temperature
    sink_temp = temp_sink + gradient_sink * weather_df[C.TEMP_AIR]

    # Calculate temperature difference
    delta_t = sink_temp - weather_df[defs.SOURCE_MAP[hp_source]]

    # Apply temperature offset for ground and water source heat pumps
    if hp_source in temp_offset_types:
        delta_t -= temp_offset

    # Apply COP formula and correction factor
    cop_values = _apply_cop_formula(delta_t, coeffs) * correction_factor

    return cop_values


def _apply_cop_formula(delta_t, coeffs):
    """Apply the quadratic COP formula with the provided coefficients.

    Args:
        delta_t (pd.Series): Temperature difference series
        coeffs (dict): Coefficients for the quadratic formula with keys 'a', 'b', and 'c'

    Returns:
        pd.Series: COP values calculated using the quadratic formula

    Raises:
        KeyError: If coeffs is missing any of the required keys
    """
    # Ensure all required keys are present
    if not all(key in coeffs for key in ["a", "b", "c"]):
        raise KeyError("Coefficients must contain keys 'a', 'b', and 'c'")

    # Apply quadratic formula: c + b*x + a*x²
    return coeffs["c"] + coeffs["b"] * delta_t + coeffs["a"] * delta_t**2
