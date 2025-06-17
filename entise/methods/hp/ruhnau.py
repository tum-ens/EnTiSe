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

import numpy as np
import pandas as pd

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import Types
from entise.core.base import Method

logger = logging.getLogger(__name__)

# Define heat pump source types
HP_AIR = "ASHP"  # Air Source Heat Pump
HP_SOIL = "GSHP"  # Ground Source Heat Pump
HP_WATER = "WSHP"  # Water Source Heat Pump
SOURCES = [HP_AIR, HP_SOIL, HP_WATER]

# Define default heat pump parameters for each technology
HP_SYSTEM = {
    HP_AIR: {
        "a": 0.0005,
        "b": -0.09,
        "c": 6.08,
    },
    HP_SOIL: {
        "a": 0.0012,
        "b": -0.21,
        "c": 10.29,
    },
    HP_WATER: {
        "a": 0.0012,
        "b": -0.20,
        "c": 9.97,
    },
}

# Define heat sink types
FLOOR = "floor"
RADIATOR = "radiator"
WATER = "water"

# Define default temperature parameters for each sink type
SINKS = {
    FLOOR: {O.TEMP_SINK: 30, O.GRADIENT_SINK: -0.5},
    RADIATOR: {O.TEMP_SINK: 40, O.GRADIENT_SINK: -1.0},
    WATER: {O.TEMP_SINK: 50, O.GRADIENT_SINK: 0},
}

# Temperature column names
SOURCE_MAP = {HP_AIR: C.TEMP, HP_SOIL: C.TEMP_SOIL, HP_WATER: C.TEMP_WATER_GROUND}

# Default correction factor
DEFAULT_CORRECTION_FACTOR = 0.85


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
        f"{Types.HP}_{Types.HEATING}_avg": "average heating COP value",
        f"{Types.HP}_{Types.HEATING}_min": "minimum heating COP value",
        f"{Types.HP}_{Types.HEATING}_max": "maximum heating COP value",
        f"{Types.HP}_{Types.DHW}_avg": "average DHW COP value",
        f"{Types.HP}_{Types.DHW}_min": "minimum DHW COP value",
        f"{Types.HP}_{Types.DHW}_max": "maximum DHW COP value",
    }
    output_timeseries = {
        f"{Types.HP}_{Types.HEATING}": "heating COP time series",
        f"{Types.HP}_{Types.DHW}": "DHW COP time series",
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
            cop_coefficients (dict, optional): Custom coefficients for COP calculation

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
        )

        # Get input data with validation
        processed_obj, processed_data, custom_coefficients = self._get_input_data(
            processed_obj, processed_data, cop_coefficients
        )

        # Calculate heating COP time series
        heating_cop_series = self._calculate_heating_cop_series(processed_obj, processed_data, custom_coefficients)

        # Calculate DHW COP time series
        dhw_cop_series = self._calculate_dhw_cop_series(processed_obj, processed_data, custom_coefficients)

        # Prepare output
        summary = {
            f"{Types.HP}_{Types.HEATING}_avg": heating_cop_series.mean().round(2),
            f"{Types.HP}_{Types.HEATING}_min": heating_cop_series.min().round(2),
            f"{Types.HP}_{Types.HEATING}_max": heating_cop_series.max().round(2),
            f"{Types.HP}_{Types.DHW}_avg": dhw_cop_series.mean().round(2),
            f"{Types.HP}_{Types.DHW}_min": dhw_cop_series.min().round(2),
            f"{Types.HP}_{Types.DHW}_max": dhw_cop_series.max().round(2),
        }

        timeseries = pd.DataFrame(
            {f"{Types.HP}_{Types.HEATING}": heating_cop_series, f"{Types.HP}_{Types.DHW}": dhw_cop_series},
            index=processed_data[O.WEATHER].index,
        )

        return {
            "summary": summary,
            "timeseries": timeseries,
        }

    def _get_input_data(self, obj, data, custom_coefficients=None):
        """Process and validate input parameters.

        Args:
            obj (dict): Dictionary with heat pump parameters
            data (dict): Dictionary with input data
            custom_coefficients (dict, optional): Custom coefficients for COP calculation

        Returns:
            tuple: Processed object, data, and custom coefficients

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Get weather data
        weather_df = self.get_with_backup(data, O.WEATHER)
        if weather_df is None:
            raise ValueError("Weather data is required but not provided")

        # Ensure weather data has the right format
        weather_df = self._prepare_weather_data(weather_df)
        data[O.WEATHER] = weather_df

        # Check if a system configuration is provided
        hp_system = self.get_with_backup(obj, O.HP_SYSTEM)

        # Process heat pump parameters
        hp_source = self.get_with_backup(obj, O.HP_SOURCE)
        hp_sink = self.get_with_backup(obj, O.HP_SINK)
        hp_temp = self.get_with_backup(obj, O.TEMP_SINK)
        gradient_sink = self.get_with_backup(obj, O.GRADIENT_SINK)
        temp_water = self.get_with_backup(obj, O.TEMP_WATER)
        correction_factor = self.get_with_backup(obj, O.CORRECTION_FACTOR, DEFAULT_CORRECTION_FACTOR)

        # If a system configuration is provided, use it to set the heat pump parameters
        if hp_system is not None:
            try:
                # If hp_system is a string, assume it's a predefined system
                if isinstance(hp_system, str) and hp_system == "system":
                    # Use the default HP_SYSTEM parameters
                    logger.info("Using predefined system configuration")
                    # No need to do anything, as we'll use the HP_SYSTEM constant
                elif isinstance(hp_system, dict):
                    # Use the provided system configuration
                    logger.info("Using provided system configuration")
                    if "heat_pump_parameters" in hp_system:
                        params = hp_system["heat_pump_parameters"]
                        hp_source = params.get("hp_source", hp_source)
                        # If coefficients are provided, use them as custom coefficients
                        if all(k in params for k in ["a", "b", "c"]):
                            custom_coefficients = {"a": params.get("a"), "b": params.get("b"), "c": params.get("c")}

                    if "heat_sink_parameters" in hp_system:
                        sink_params = hp_system["heat_sink_parameters"]
                        if "heating" in sink_params:
                            heating = sink_params["heating"]
                            hp_sink = heating.get("hp_sink", hp_sink)
                            hp_temp = heating.get("temp_sink", hp_temp)
                            gradient_sink = heating.get("gradient", gradient_sink)

                        if "dhw" in sink_params:
                            dhw = sink_params["dhw"]
                            temp_water = dhw.get("temp_sink", temp_water)
                else:
                    logger.warning(f"Unknown system configuration format: {hp_system}")
            except Exception as e:
                logger.error(f"Error processing system configuration: {e}")

        # If temp_water is not provided, use a default value from SINKS
        if temp_water is None:
            temp_water = SINKS[WATER][O.TEMP_SINK]
            logger.warning(f"DHW temperature not provided, using default value of {temp_water}°C")

        # If gradient_sink is not provided, use the default value from SINKS
        if gradient_sink is None and hp_sink is not None:
            gradient_sink = SINKS[hp_sink][O.GRADIENT_SINK]
            logger.info(f"Gradient not provided, using default value of {gradient_sink} for {hp_sink}")

        # Validate parameters if custom coefficients are not provided
        if custom_coefficients is None:
            if hp_source is not None and hp_source not in SOURCES:
                raise ValueError(f"Invalid heat pump source: {hp_source}. Must be one of {SOURCES}")

            if hp_sink is not None and hp_sink not in SINKS:
                raise ValueError(f"Invalid heat sink: {hp_sink}. Must be one of {list(SINKS.keys())}")

            # Check if the temperature is valid for the selected sink
            if hp_sink is not None and hp_temp is not None:
                valid_temps = SINKS[hp_sink]["temp_0"]
                if hp_temp not in valid_temps:
                    logger.warning(
                        f"Temperature {hp_temp} not in standard values {valid_temps} for {hp_sink}. "
                        f"Using the closest value."
                    )
                    hp_temp = min(valid_temps, key=lambda x: abs(x - hp_temp))

        # Update object with processed values
        obj[O.HP_SOURCE] = hp_source
        obj[O.HP_SINK] = hp_sink
        obj[O.TEMP_SINK] = hp_temp
        obj[O.GRADIENT_SINK] = gradient_sink
        obj[O.TEMP_WATER] = temp_water
        obj[O.CORRECTION_FACTOR] = correction_factor

        return obj, data, custom_coefficients

    def _prepare_weather_data(self, weather_df):
        """Prepare weather data for COP calculation.

        Args:
            weather_df (pd.DataFrame): Weather data

        Returns:
            pd.DataFrame: Processed weather data with required temperature columns
        """
        # Make a copy to avoid modifying the original
        weather_df = weather_df.copy()

        # Ensure datetime index
        if "datetime" in weather_df.columns:
            weather_df.index = pd.to_datetime(weather_df["datetime"])
        else:
            weather_df.index = pd.to_datetime(weather_df.index)

        # Map temperature column names if they exist with different names
        column_mapping = {}

        # Check for air temperature column
        if C.TEMP not in weather_df.columns:
            if "temperature" in weather_df.columns:
                column_mapping["temperature"] = C.TEMP
                logger.info("Using 'temperature' column as air temperature")
            else:
                # If no air temperature column, add a placeholder (this will likely cause errors later)
                logger.warning("No air temperature column found in weather data")
                weather_df[C.TEMP] = np.nan

        # Check for soil temperature column
        if C.TEMP_SOIL not in weather_df.columns:
            if "soil_temperature" in weather_df.columns:
                column_mapping["soil_temperature"] = C.TEMP_SOIL
                logger.info("Using 'soil_temperature' column as soil temperature")
            else:
                # Calculate soil temperature from air temperature
                logger.info("Calculating soil temperature from air temperature")
                air_temp_col = C.TEMP if C.TEMP in weather_df.columns else "temperature"
                weather_df["rolling_temp"] = weather_df[air_temp_col].rolling(window=24, min_periods=1).mean()
                weather_df[C.TEMP_SOIL] = weather_df["rolling_temp"].apply(self._calc_soil_temp)
                weather_df.drop(columns=["rolling_temp"], inplace=True)

        # Check for groundwater temperature column
        if C.TEMP_WATER_GROUND not in weather_df.columns:
            if "groundwater_temperature" in weather_df.columns:
                column_mapping["groundwater_temperature"] = C.TEMP_WATER_GROUND
                logger.info("Using 'groundwater_temperature' column as groundwater temperature")
            else:
                # Set default groundwater temperature
                logger.info("Setting default groundwater temperature to 10°C")
                weather_df[C.TEMP_WATER_GROUND] = 10

        # Rename columns if needed
        if column_mapping:
            weather_df.rename(columns=column_mapping, inplace=True)

        return weather_df

    def _calc_soil_temp(self, t_avg_d):
        """Calculate soil temperature based on average air temperature.

        Based on "WP Monitor" Feldmessung von Wärmepumpenanlagen, Frauenhofer ISE, 2014.

        Args:
            t_avg_d (float): Average air temperature

        Returns:
            float: Calculated soil temperature
        """
        t_soil = -0.0003 * t_avg_d**3 + 0.0086 * t_avg_d**2 + 0.3047 * t_avg_d + 5.0647
        return t_soil

    def _calculate_heating_cop_series(self, obj, data, custom_coefficients=None):
        """Calculate heating COP time series based on temperature differences.

        Args:
            obj (dict): Dictionary with heat pump parameters
            data (dict): Dictionary with input data
            custom_coefficients (dict, optional): Custom coefficients for COP calculation

        Returns:
            pd.Series: Heating COP time series
        """
        weather_df = data[O.WEATHER]
        hp_source = obj[O.HP_SOURCE]
        hp_sink = obj[O.HP_SINK]
        hp_temp = obj[O.TEMP_SINK]
        gradient_sink = obj[O.GRADIENT_SINK]
        correction_factor = obj[O.CORRECTION_FACTOR]

        if custom_coefficients is not None:
            # Use custom coefficients for COP calculation
            return self._calculate_cop_with_custom_coefficients(weather_df, custom_coefficients, correction_factor)
        elif hp_source is not None and hp_sink is not None and hp_temp is not None:
            # Use standard model with heat pump type and sink
            return self._calculate_cop_with_standard_model(
                weather_df, hp_source, hp_sink, hp_temp, correction_factor, gradient_sink
            )
        else:
            # If parameters are missing, use default values
            logger.warning("Missing parameters for heating COP calculation, using default values")
            return self._calculate_cop_with_standard_model(
                weather_df,
                HP_AIR,  # Default to air source
                RADIATOR,  # Default to radiator
                40,  # Default to 40°C
                correction_factor,
            )

    def _calculate_dhw_cop_series(self, obj, data, custom_coefficients=None):
        """Calculate DHW COP time series based on temperature differences.

        Args:
            obj (dict): Dictionary with heat pump parameters
            data (dict): Dictionary with input data
            custom_coefficients (dict, optional): Custom coefficients for COP calculation

        Returns:
            pd.Series: DHW COP time series
        """
        weather_df = data[O.WEATHER]
        hp_source = obj[O.HP_SOURCE]
        temp_water = obj[O.TEMP_WATER]
        correction_factor = obj[O.CORRECTION_FACTOR]
        # For DHW, we use the gradient from SINKS[WATER]
        gradient_sink = SINKS[WATER][O.GRADIENT_SINK]

        if custom_coefficients is not None:
            # Use custom coefficients for COP calculation
            return self._calculate_cop_with_custom_coefficients(weather_df, custom_coefficients, correction_factor)
        elif hp_source is not None and temp_water is not None:
            # Use standard model with heat pump type and water as sink
            return self._calculate_cop_with_standard_model(
                weather_df,
                hp_source,
                WATER,  # Always use 'water' as sink type for DHW
                temp_water,
                correction_factor,
                gradient_sink,
            )
        else:
            # If parameters are missing, use default values
            logger.warning("Missing parameters for DHW COP calculation, using default values")
            return self._calculate_cop_with_standard_model(
                weather_df,
                HP_AIR,  # Default to air source
                WATER,  # Always use 'water' as sink type for DHW
                50,  # Default to 50°C
                correction_factor,
                gradient_sink,
            )

    def _calculate_cop_with_standard_model(
        self, weather_df, hp_source, hp_sink, hp_temp, correction_factor, gradient_sink=None
    ):
        """Calculate COP using the standard model based on heat pump type and sink.

        Args:
            weather_df (pd.DataFrame): Weather data
            hp_source (str): Heat pump source type
            hp_sink (str): Heat sink type
            hp_temp (float): Temperature setting
            correction_factor (float): Efficiency correction factor
            gradient_sink (float, optional): Gradient for sink temperature calculation.
                If None, uses the default gradient from SINKS.

        Returns:
            pd.Series: COP time series
        """
        # Use provided gradient or get default from SINKS
        if gradient_sink is None:
            gradient_sink = SINKS[hp_sink][O.GRADIENT_SINK]

        # Calculate sink temperature based on ambient temperature
        sink_temp = hp_temp + gradient_sink * weather_df[C.TEMP]

        # Calculate temperature difference
        delta_t = sink_temp - weather_df[SOURCE_MAP[hp_source]]

        # Apply temperature offset for ground and water source heat pumps
        if hp_source in [HP_SOIL, HP_WATER]:
            delta_t -= 5

        # Apply COP formula based on heat pump type
        cop_values = self._apply_cop_formula(hp_source, delta_t, custom_coeffs=None)

        # Apply correction factor
        cop_values = cop_values * correction_factor

        return cop_values

    def _calculate_cop_with_custom_coefficients(self, weather_df, coefficients, correction_factor):
        """Calculate COP using custom coefficients.

        This function is a wrapper around _apply_cop_formula that handles the extraction
        of temperature data from the weather DataFrame and applies the correction factor
        to the resulting COP values.

        Args:
            weather_df (pd.DataFrame): Weather data containing temperature information
            coefficients (dict): Custom coefficients for the quadratic formula.
                Should contain keys 'a', 'b', and 'c'.
            correction_factor (float): Efficiency correction factor to apply to the COP values

        Returns:
            pd.Series: COP time series calculated with custom coefficients and correction factor
        """
        # Use air temperature as default
        temp_diff = weather_df[C.TEMP]

        # Apply COP formula using the _apply_cop_formula method with custom coefficients
        cop_values = self._apply_cop_formula(None, temp_diff, custom_coeffs=coefficients)

        # Apply correction factor
        cop_values = cop_values * correction_factor

        return cop_values

    def _apply_cop_formula(self, hp_source, delta_t, custom_coeffs=None):
        """Apply the appropriate COP formula based on heat pump type or custom coefficients.

        This function calculates COP values using a quadratic formula of the form:
        COP = c + b * delta_t + a * delta_t^2

        The coefficients (a, b, c) are either taken from the predefined HP_SYSTEM dictionary
        based on the heat pump source type, or from the provided custom_coeffs dictionary.

        Args:
            hp_source (str): Heat pump source type (used if custom_coeffs is None)
            delta_t (pd.Series): Temperature difference series
            custom_coeffs (dict, optional): Custom coefficients for the quadratic formula.
                Should contain keys 'a', 'b', and 'c'. If provided, hp_source is ignored.

        Returns:
            pd.Series: COP values calculated using the quadratic formula
        """
        if custom_coeffs is not None:
            # Use custom coefficients
            a = custom_coeffs.get("a", 0.0005)
            b = custom_coeffs.get("b", -0.09)
            c = custom_coeffs.get("c", 6.08)
            return c + b * delta_t + a * delta_t**2
        elif hp_source in HP_SYSTEM:
            # Use predefined coefficients for the heat pump source
            coeffs = HP_SYSTEM[hp_source]
            return coeffs["c"] + coeffs["b"] * delta_t + coeffs["a"] * delta_t**2
        else:
            raise ValueError(f"Unknown heat pump source: {hp_source}")
