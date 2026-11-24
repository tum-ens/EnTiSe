import logging
from collections.abc import Mapping

import pandas as pd

from entise._optional import requires_extra
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.constants import UnitConversion as UConv
from entise.core._time import RUN_TIME_KEY, output_index, regular_step, run_time
from entise.core.base import Method

logger = logging.getLogger(__name__)

_WIND_SPEED = C.WIND_SPEED.split("[")[0]


def _warn_wind_speed_without_height(table: str):
    """Return a check that warns, once per run and table, about wind speed columns without a height."""

    def check(weather: pd.DataFrame) -> list:
        _, info = Method._obtain_weather_info(weather)
        columns = [col for col, parsed in info.items() if parsed["name"] == _WIND_SPEED and parsed["height"] is None]
        for column in columns:
            logger.warning(
                "Table '%s': column '%s' has no height; WPLib reads it at %d m. Name the column "
                "'%s@<height>m' to give its height.",
                table,
                column,
                WIND_SPEED_HEIGHT,
                C.WIND_SPEED,
            )
        return columns

    return check


# Default values for optional keys
DEFAULT_TURBINE_TYPE = "SWT130/3600"  # turbine type as in register of windpowerlib
DEFAULT_HUB_HEIGHT = 135  # in m
DEFAULT_POWER = 1  # in W
ROUGHNESS_LENGTH = 0.15
# Height of a wind speed column that names none: the standard height of wind measurements.
WIND_SPEED_HEIGHT = 10

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
    """Wind power generation using windpowerlib model chain (AC power output).

    This module integrates the windpowerlib package to compute wind turbine power
    from meteorological inputs and turbine metadata. It constructs a windpowerlib
    ModelChain with a chosen turbine (by type/name), corrects wind speed to hub height
    (if required), and converts met data (wind speed, temperature, pressure) into hub‑height
    conditions and electrical output.

    Key capabilities:
    - Accept a site weather DataFrame and turbine parameters (type, hub height, rated power).
    - Use windpowerlib's power curve–based ModelChain for robust, transparent calculations.
    - Return AC power time series along with summary KPIs (max generation, full‑load hours).

    Reference: windpowerlib — https://windpowerlib.readthedocs.io/
    """

    types = [Types.WIND]
    name = "wplib"
    required_keys = [O.WEATHER]
    optional_keys = {
        O.POWER: DEFAULT_POWER,
        O.TURBINE_TYPE: DEFAULT_TURBINE_TYPE,
        O.HUB_HEIGHT: DEFAULT_HUB_HEIGHT,
        # ModelChain settings that replace those in MODELCHAIN_PARAMS: a mapping, or the name of one in data.
        O.WIND_MODEL: None,
        O.TIMEZONE: None,
    }
    required_data = [O.WEATHER]
    optional_data = [O.WIND_MODEL]
    output_summary = {
        f"{Types.WIND}{SEP}{C.GENERATION}": "total wind power generation",
        f"{Types.WIND}{SEP}{O.GEN_MAX}": "maximum wind power generation",
        f"{Types.WIND}{SEP}{C.FLH}": "full load hours",
    }
    output_timeseries = {
        f"{C.GENERATION}_{Types.WIND}": "wind power generation",
    }

    def generate(
        self,
        obj: dict | None = None,
        data: dict | None = None,
        results: dict | None = None,
        ts_type: str = Types.WIND,
        **inputs,
    ) -> dict:
        """Generate wind power time series based on input parameters and weather data.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the wind power generation time series,
        and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing wind turbine parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            results (dict, optional): Dictionary with results from previously generated time series
            ts_type (str, optional): Time series type to generate. Defaults to Types.WIND.
            **inputs: Any input of the method, by its key or by its Python name (the key
                without its unit, such as ``area`` for ``area[m2]``). Keywords override the
                same keys in ``obj``. A table input, such as ``weather``, can be a
                DataFrame. For example:

                - ``weather`` (DataFrame or str): Weather data with wind speed and direction.
                - ``power`` (``power[W]``): System power rating in W.
                - ``turbine_type`` (str): Type of wind turbine.
                - ``hub_height`` (``hub_height[m]``): Hub height of the turbine in m.

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
            >>> obj = {"power[W]": 5000, "weather": "weather"}
            >>> data = {"weather": weather_df}  # DataFrame with wind data
            >>> result = windlib.generate(obj=obj, data=data)
            >>> summary = result["summary"]
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(obj, data, ts_type, **inputs)

        # Continue with existing implementation
        # Read the tables once for this call; every helper below reuses this run.
        processed_data = {**processed_data, RUN_TIME_KEY: run_time(processed_data)}
        site_obj, run_data = processed_obj, processed_data
        processed_obj, processed_data = self.get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)
        ts.index = output_index(processed_data[O.WEATHER].index, site_obj, run_data)

        logger.debug(f"[WIND windlib]: Generating {ts_type} data")

        return self._format_output(processed_obj, processed_data, ts)

    def process_weather_data(self, weather_data):
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
        weather, info = self._obtain_weather_info(weather)

        weather = self._create_multiindex(weather, info)

        weather = weather.rename(
            columns={
                C.TEMP_AIR.split("[")[0]: "temperature",
                C.SURFACE_AIR_PRESSURE.split("[")[0]: "pressure",
                C.WIND_SPEED.split("[")[0]: "wind_speed",
                C.WIND_DIRECTION.split("[")[0]: "wind_direction",
                C.ROUGHNESS_LENGTH.split("[")[0]: "roughness_length",
            },
            level=0,
        )

        weather["temperature"] += UConv.CELSIUS2KELVIN.value

        return weather

    @staticmethod
    def _create_multiindex(df, info):
        """Reshape the dataframe to have a multi-index for columns as needed by windpowerlib.

        Args:
            df (pd.DataFrame): Weather dataframe to reshape.
            info (dict): Information dictionary with column information including height.

        Returns:
            pd.DataFrame: Reshaped dataframe with multi-index columns.
        """
        df.index.name = None

        # Create arrays for MultiIndex using vectorized operations
        names = [info.get(col, {}).get("name", col) for col in df.columns]
        heights = [info.get(col, {}).get("height") for col in df.columns]
        # A wind speed without a height is taken at 10 m; other quantities at 0 m, as before.
        heights = [
            int(height) if height is not None else WIND_SPEED_HEIGHT if name == _WIND_SPEED else 0
            for name, height in zip(names, heights, strict=True)
        ]
        arrays = [names, heights]  # variable_name, height

        df.columns = pd.MultiIndex.from_arrays(arrays, names=["variable_name", "height"])

        return df

    def get_input_data(self, obj, data, method_type=Types.WIND):
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
        obj_out = self.get_relevant_objects(obj, method_type)

        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        run = run_time(data)
        data_out = {
            O.WEATHER: run.table(weather_key) if weather_key in run else None,
            O.WIND_MODEL: modelchain_params(obj_out.get(O.WIND_MODEL), data),
        }

        # Process weather data. The run returns a copy indexed by UTC instants.
        if data_out[O.WEATHER] is not None:
            regular_step(data_out[O.WEATHER].index, table=weather_key)
            run.derived(weather_key, "wplib:wind speed without height", _warn_wind_speed_without_height(weather_key))
            data_out[O.WEATHER] = self.process_weather_data(data_out[O.WEATHER])
        else:
            logger.error("[WIND windlib]: No weather data")
            raise Exception(f"{O.WEATHER} not available")

        return obj_out, data_out

    @staticmethod
    def _format_output(processed_obj, processed_data, ts):
        timestep = regular_step(processed_data[O.WEATHER].index, table=O.WEATHER).total_seconds()
        summary = {
            f"{Types.WIND}{SEP}{C.GENERATION}": (ts[O.POWER].sum() * timestep / 3600).round().astype(int),
            f"{Types.WIND}{SEP}{O.GEN_MAX}": ts[O.POWER].max().round().astype(int),
            f"{Types.WIND}{SEP}{C.FLH}": (ts[O.POWER].sum() * timestep / 3600 / processed_obj[O.POWER])
            .round()
            .astype(int),
        }

        ts = ts.rename(columns={O.POWER: f"{Types.WIND}{SEP}{C.POWER}"})

        return {
            "summary": summary,
            "timeseries": ts,
        }


def modelchain_params(wind_model, data: dict) -> dict:
    """Return the windpowerlib ModelChain settings for an object.

    Args:
        wind_model: The object's ``wind_model``: ``None`` for the defaults in
            ``MODELCHAIN_PARAMS``, a mapping of settings that replace them, such as
            ``{"wind_speed_model": "hellman"}``, or the name of such a mapping in ``data``.
        data (dict): The input data.

    Raises:
        ValueError: If ``wind_model`` names no mapping, or sets an unknown setting.
    """
    if wind_model is None:
        return dict(MODELCHAIN_PARAMS)
    settings = data.get(wind_model) if isinstance(wind_model, str) else wind_model
    if not isinstance(settings, Mapping):
        raise ValueError(
            f"'{O.WIND_MODEL}' must be a mapping of ModelChain settings or the name of one in the data; "
            f"got {wind_model!r}."
        )
    unknown = sorted(set(settings) - set(MODELCHAIN_PARAMS))
    if unknown:
        raise ValueError(
            f"Unknown ModelChain settings in '{O.WIND_MODEL}': {unknown}. Known: {sorted(MODELCHAIN_PARAMS)}."
        )
    return {**MODELCHAIN_PARAMS, **settings}


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
          defined in MODELCHAIN_PARAMS, updated by the object's ``wind_model``.
        - The output power is scaled by the system power rating.
    """
    with requires_extra("wind-wplib", "windpowerlib", feature="Wind method 'wplib'"):
        from windpowerlib import ModelChain, WindTurbine

    # Get objects
    power = obj[O.POWER]
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
    params = data.get(O.WIND_MODEL, MODELCHAIN_PARAMS)
    try:
        mc = ModelChain(turbine, **params).run_model(weather)
    except IndexError as err:
        # windpowerlib interpolates between heights and fails without a second one.
        raise ValueError(
            f"windpowerlib cannot run the ModelChain settings {params} on this weather: an "
            "'interpolation_extrapolation' model needs its variable at two or more heights."
        ) from err

    # Get output power
    power_output = mc.power_output

    # Set time index to origin timestamp
    power_output.index.name = C.DATETIME
    power_output.name = O.POWER

    # Create DataFrame and rename column
    df = pd.DataFrame(power_output, columns=[O.POWER])

    # Calculate and round power (normalize by nominal power)
    df[O.POWER] = df[O.POWER] / turbine.nominal_power * power
    df = df.round(3)

    return df
