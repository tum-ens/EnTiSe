import logging

import numpy as np
import pandas as pd

from entise._optional import requires_extra
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import RUN_TIME_KEY, output_index, regular_step, run_time
from entise.core.base import Method

logger = logging.getLogger(__name__)

# Default values for optional keys
POWER = 1  # W
AZIMUTH = 0  # º
TILT = 0  # º
PV_ARRAYS = dict(
    module_parameters=dict(pdc0=1, gamma_pdc=-0.004),
    temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3),
)
PV_INVERTER = dict(pdc0=3)


class PVLib(Method):
    """PV generation using pvlib-python model chain (AC power output).

    Purpose and scope:
    - Builds a simple pvlib.pvsystem ModelChain given site latitude/longitude,
      panel tilt/azimuth, array and inverter parameters, and meteorological
      inputs (GHI/DNI/DHI, temperature, wind). Produces AC power time series and
      summary KPIs (max generation, full-load hours).

    Notes:
    - Column naming for inputs follows EnTiSe conventions (Columns.*). Height
      suffixes like ``@2m`` are automatically stripped where applicable.
    - When only GHI is available, pvlib can approximate plane-of-array irradiance
      using transposition models; results improve with DNI/DHI present.

    Reference:
    - pvlib-python documentation: https://pvlib-python.readthedocs.io/
    """

    types = [Types.PV]
    name = "pvlib"
    required_keys = [O.LAT, O.LON, O.WEATHER]
    optional_keys = {
        O.POWER: POWER,
        O.AZIMUTH: AZIMUTH,
        O.TILT: TILT,
        O.ALTITUDE: None,
        O.PV_ARRAYS: None,
        O.PV_INVERTER: None,
        O.TIMEZONE: None,
    }
    required_data = [O.WEATHER]
    optional_data = [O.PV_ARRAYS, O.PV_INVERTER]
    output_summary = {
        f"{Types.PV}{SEP}{C.GENERATION}": "total PV generation",
        f"{Types.PV}{SEP}{O.GEN_MAX}": "maximum PV generation",
        f"{Types.PV}{SEP}{C.FLH}": "full load hours",
    }
    output_timeseries = {
        f"{C.GENERATION}_{Types.PV}": "PV generation",
    }

    def generate(
        self,
        obj: dict | None = None,
        data: dict | None = None,
        results: dict | None = None,
        ts_type: str = Types.PV,
        **inputs,
    ) -> dict:
        """Generate PV power time series based on input parameters and weather data.

        This method implements the abstract generate method from the Method base class.
        It processes the input parameters, calculates the PV generation time series,
        and returns both the time series and summary statistics.

        Args:
            obj (dict, optional): Dictionary containing PV system parameters. Defaults to None.
            data (dict, optional): Dictionary containing input data. Defaults to None.
            results (dict, optional): Dictionary with results from previously generated time series
            ts_type (str, optional): Time series type to generate. Defaults to Types.PV.
            **inputs: Any input of the method, by its key or by its Python name (the key
                without its unit, such as ``area`` for ``area[m2]``). Keywords override the
                same keys in ``obj``. A table input, such as ``weather``, can be a
                DataFrame. For example:

                - ``latitude`` and ``longitude`` (``[degree]``): Geographic latitude and longitude in degrees.
                - ``weather`` (DataFrame or str): Weather data with solar radiation.
                - ``power`` (``power[W]``): System power rating in W.
                - ``azimuth`` (``azimuth[degree]``): Panel azimuth in degrees (0=North, 90=East, 180=South,
                  270=West).
                - ``tilt`` (``tilt[degree]``): Panel tilt in degrees (0=horizontal, 90=vertical).
                - ``altitude`` (``altitude[m]``): Site altitude in m.
                - ``pv_arrays`` (dict or str): PV array configuration parameters, or the
                  name of a table in ``data`` that holds them. Without one, the arrays use
                  ``pdc0=1`` and ``gamma_pdc=-0.004``.
                - ``pv_inverter`` (dict or str): PV inverter configuration parameters, or the
                  name of a table in ``data`` that holds them. An object that names none uses
                  the table under ``data["pv_inverter"]`` if there is one, otherwise ``pdc0=3``.

                A table that an object names must be in ``data``; otherwise the method
                raises a ``ValueError``.

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
            >>> obj = {"latitude[degree]": 48.1, "longitude[degree]": 11.6, "power[W]": 5000}
            >>> data = {"weather": weather_df}  # DataFrame with solar radiation data
            >>> result = pvlib.generate(obj=obj, data=data)
            >>> summary = result["summary"]
            >>> timeseries = result["timeseries"]
        """
        # Process keyword arguments
        processed_obj, processed_data = self._process_kwargs(obj, data, ts_type, **inputs)

        # Continue with existing implementation
        # Read the tables once for this call; every helper below reuses this run.
        processed_data = {**processed_data, RUN_TIME_KEY: run_time(processed_data)}
        site_obj, run_data = processed_obj, processed_data
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)
        ts.index = output_index(ts.index, site_obj, run_data)

        logger.debug(f"[PV pvlib]: Generating {ts_type} data")

        return self._format_output(ts, processed_obj, processed_data)

    def _get_input_data(self, obj, data, method_type=Types.PV):
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
        obj_out = self.get_relevant_objects(obj, method_type)
        weather_key = Method.get_with_method_backup(obj, O.WEATHER, method_type, O.WEATHER)
        run = run_time(data)
        data_out = {
            O.WEATHER: run.table(weather_key) if weather_key in run else None,
            O.PV_ARRAYS: Method.named_table(obj_out, data, O.PV_ARRAYS, PV_ARRAYS),
            # An object that names no inverter uses the table under the key itself, then the default.
            O.PV_INVERTER: Method.named_table(obj_out, data, O.PV_INVERTER, PV_INVERTER, fixed_key=O.PV_INVERTER),
        }

        # Fill missing data
        if obj_out.get(O.ALTITUDE) is None:
            with requires_extra("pv-pvlib", "pvlib"):
                import pvlib

            obj_out[O.ALTITUDE] = pvlib.location.lookup_altitude(obj_out.get(O.LAT), obj_out.get(O.LON))

        if data_out[O.WEATHER] is not None:
            # The run returns a copy indexed by UTC instants; each row is the interval [t, t + step).
            weather = self._strip_weather_height(data_out[O.WEATHER])
            weather = weather.loc[:, ~weather.columns.duplicated()].copy()
            weather = weather.rename(
                columns={
                    f"{C.SOLAR_GHI}": "ghi",
                    f"{C.SOLAR_DNI}": "dni",
                    f"{C.SOLAR_DHI}": "dhi",
                    f"{C.TEMP_AIR}": "temp_air",
                    f"{C.WIND_SPEED}": "wind_speed",
                },
            )
            regular_step(weather.index, table=weather_key)
            if C.SOLAR_GHI in data_out[O.WEATHER].columns and obj_out.get(O.LON) is not None:
                run.check_irradiance(weather_key, float(obj_out[O.LON]))
            data_out[O.WEATHER] = weather
        else:
            logger.error("[PV pvlib]: No weather data")
            raise Exception(f"{O.WEATHER} not available")

        # Sanity checks
        if not 0 <= obj_out[O.AZIMUTH] <= 360:
            logger.warning(f"Azimuth value {obj_out[O.AZIMUTH]} outside normal range [0-360]")
        if not 0 <= obj_out[O.TILT] <= 90:
            logger.warning(f"Tilt value {obj_out[O.TILT]} outside normal range [0-90]")

        return obj_out, data_out

    @staticmethod
    def _format_output(ts, processed_obj, processed_data):
        timestep = regular_step(processed_data[O.WEATHER].index, table=O.WEATHER).total_seconds()
        # Scalars, so the summary table has numeric columns (a one-column frame's sum is a Series).
        power = ts["p_mp"]
        energy_wh = power.sum() * timestep / 3600
        summary = {
            f"{Types.PV}{SEP}{C.GENERATION}": int(np.round(energy_wh)),
            f"{Types.PV}{SEP}{O.GEN_MAX}": int(np.round(power.max())),
            f"{Types.PV}{SEP}{C.FLH}": int(np.round(energy_wh / processed_obj[O.POWER])),
        }

        ts = ts.rename(columns={"p_mp": f"{Types.PV}{SEP}{C.POWER}"})

        return {
            "summary": summary,
            "timeseries": ts,
        }


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
    with requires_extra("pv-pvlib", "pvlib"):
        import pvlib
        from pvlib import pvsystem

    lat = obj[O.LAT]
    lon = obj[O.LON]
    power = obj[O.POWER]
    tilt = obj[O.TILT]

    # Get data
    df_weather = data[O.WEATHER]
    pv_arrays = data[O.PV_ARRAYS]
    pv_inverter = data[O.PV_INVERTER]

    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=altitude)

    # Create the pv system
    array = pvsystem.Array(pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth), **pv_arrays)
    system = pvsystem.PVSystem(arrays=array, inverter_parameters=pv_inverter)
    mc = pvlib.modelchain.ModelChain(system, loc, aoi_model="physical", spectral_model="no_loss")

    # Each row is the mean over [t, t + step), so the model sees the sun at the interval midpoint.
    axis = df_weather.index
    step = axis[1] - axis[0]
    mc.run_model(df_weather.set_axis(axis + step / 2))

    return pd.DataFrame(mc.results.ac.to_numpy() * power, index=axis, columns=["p_mp"])
