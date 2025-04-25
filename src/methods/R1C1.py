__author__ = "Markus Doepfert"
__license__ = "MIT"
__maintainer__ = "Markus Doepfert"
__email__ = "markus.doepfert@tum.de"
__status__ = "Stable"
__date__ = "2025-04-14"
__credits__ = []
__description__ = "A module for generating HVAC (heating, ventilation & air-conditioning) timeseries using a 1R1C thermal model."
__url__ = ""
__dependencies__ = ["numba", "numpy", "pandas", "pvlib"]

import logging
import time

from numba import njit
import numpy as np
import pandas as pd
import pvlib

from src.core.base import TimeSeriesMethod
from src.core.registry import Meta
from src.utils.decorators import supported_types
from src.constants import Columns as C, Keys as K, Objects as O, Types

logger = logging.getLogger(__name__)

# Possible expansions:
# - Include thermal inertia to control how quickly the heating system reacts to temperature changes
#   ((1. - self.thermal_inertia) * min(self.sh_power[iii], self.sh_powermax)
#   + self.sh_power[iii - 1] * self.thermal_inertia)

# Default values for optional keys
DEFAULT_ACTIVE_COOLING = True
DEFAULT_ACTIVE_HEATING = True
DEFAULT_ACTIVE_SOLAR_GAINS = True
DEFAULT_ACTIVE_INTERNAL_GAINS = True
DEFAULT_GAINS_INTERNAL = 500
DEFAULT_POWER_HEATING = np.inf
DEFAULT_POWER_COOLING = np.inf
DEFAULT_TEMP_INIT = 22
DEFAULT_TEMP_MAX = 24
DEFAULT_TEMP_MIN = 20
DEFAULT_THERMAL_INERTIA = 0
DEFAULT_VENTILATION = 0


@supported_types(Types.HVAC)
class R1C1(TimeSeriesMethod, metaclass=Meta):
    """Timeseries generation method using a 1R1C-model."""
    # Define required keys for each method
    required_keys = {O.WEATHER: O.DTYPES[O.WEATHER],
                     O.RESISTANCE: O.DTYPES[O.RESISTANCE],
                     O.CAPACITANCE: O.DTYPES[O.CAPACITANCE],
                     }
    optional_keys = {O.GAINS_INTERNAL: O.DTYPES[O.GAINS_INTERNAL],
                     O.GAINS_INTERNAL_COL: O.DTYPES[O.GAINS_INTERNAL_COL],
                     O.LAT: O.DTYPES[O.LAT],
                     O.LON: O.DTYPES[O.LON],
                     O.POWER_COOLING: O.DTYPES[O.POWER_COOLING],
                     O.POWER_HEATING: O.DTYPES[O.POWER_HEATING],
                     O.TEMP_INIT: O.DTYPES[O.TEMP_INIT],
                     O.TEMP_MAX: O.DTYPES[O.TEMP_MAX],
                     O.TEMP_MIN: O.DTYPES[O.TEMP_MIN],
                     O.VENTILATION: O.DTYPES[O.VENTILATION],
                     }
    # Define required timeseries for each method
    required_timeseries = {O.WEATHER: {K.COLS_REQUIRED: {C.DATETIME: C.DTYPES[C.DATETIME],
                                                         C.TEMP_OUT: C.DTYPES[C.TEMP_OUT],
                                                         },
                                       K.COLS_OPTIONAL: {
                                           C.SOLAR_DHI: C.DTYPES[C.SOLAR_DHI],
                                           C.SOLAR_DNI: C.DTYPES[C.SOLAR_DNI],
                                           C.SOLAR_GHI: C.DTYPES[C.SOLAR_GHI],
                                       },
                                       K.DTYPE: pd.DataFrame
                                       },
                           }
    optional_timeseries = {O.WINDOWS: {K.COLS_REQUIRED: {C.ID: C.DTYPES[C.ID],
                                                         C.AREA: C.DTYPES[C.AREA],
                                                         C.TRANSMITTANCE: C.DTYPES[C.TRANSMITTANCE],
                                                         C.ORIENTATION: C.DTYPES[C.ORIENTATION],
                                                         C.TILT: C.DTYPES[C.TILT],
                                                         C.SHADING: C.DTYPES[C.SHADING],
                                                         },
                                       K.COLS_OPTIONAL: {},
                                       K.DTYPE: pd.DataFrame},
                           O.GAINS_INTERNAL: {K.COLS_REQUIRED: {C.DATETIME: C.DTYPES[C.DATETIME],
                                                                },
                                              K.COLS_OPTIONAL: {},
                                              K.DTYPE: pd.DataFrame},
                           }
    # Define dependencies for each method
    dependencies = []
    # Available outputs (placeholders for summary and timeseries outputs)
    available_outputs = {
        K.SUMMARY: {
            f'{C.DEMAND}_{Types.HEATING}': 'total heating demand',
            f'{C.DEMAND}_{Types.COOLING}': 'total cooling demand',
        },
        K.TIMESERIES: {
            f'{C.TEMP_IN}': 'indoor temperature',
            f'{C.LOAD}_{Types.HEATING}': 'heating load',
            f'{C.LOAD}_{Types.COOLING}': 'cooling load',
        }
    }

    def generate(self, obj: dict, data: dict, ts_type: str = Types.HVAC, **kwargs) -> (dict, pd.DataFrame):
        """
        Generate HVAC timeseries using degree days.

        Parameters:
        - obj (dict): Objects metadata and parameters.
        - data (dict): Timeseries data.

        Returns:
        - pd.DataFrame: Timeseries for HVAC demand.
        """
        # Prepare inputs
        obj = self.prepare_inputs(obj, data, ts_type)

        return calculate(obj, data)


def calculate(obj: dict, data: dict) -> (dict, pd.DataFrame):
    """
    Optimized calculation of the 1R1C thermal model for a single building.
    """
    # Unpack parameters
    # Objects
    object_id = obj.get(O.ID, None)
    active_cool = obj.get(O.ACTIVE_COOLING, DEFAULT_ACTIVE_COOLING)
    active_heat = obj.get(O.ACTIVE_HEATING, DEFAULT_ACTIVE_HEATING)
    active_solar_gains = obj.get(O.ACTIVE_GAINS_SOLAR, DEFAULT_ACTIVE_SOLAR_GAINS)
    active_internal_gains = DEFAULT_ACTIVE_INTERNAL_GAINS
    internal_gains = obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL)
    lat = obj.get(O.LAT, None)
    lon = obj.get(O.LON, None)
    power_cool_max = obj.get(O.POWER_COOLING, DEFAULT_POWER_COOLING)
    power_heat_max = obj.get(O.POWER_HEATING, DEFAULT_POWER_HEATING)
    thermal_resistance = obj[O.RESISTANCE]
    thermal_capacitance = obj[O.CAPACITANCE]
    temp_init = obj.get(O.TEMP_INIT, DEFAULT_TEMP_INIT)
    temp_max = obj.get(O.TEMP_MAX, DEFAULT_TEMP_MAX)
    temp_min = obj.get(O.TEMP_MIN, DEFAULT_TEMP_MIN)
    ventilation = obj.get(O.VENTILATION, DEFAULT_VENTILATION)
    # Data
    # Weather
    weather = data[O.WEATHER]
    temp_out = weather[C.TEMP_OUT].to_numpy(dtype=np.float32)
    weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME])  # Ensure datetime format
    weather.set_index(C.DATETIME, inplace=True, drop=False)
    # Windows
    windows = data.get(O.WINDOWS, None)
    if windows is not None:
        windows = windows.loc[windows[O.ID] == object_id]
        windows = windows if not windows.empty else None
    # Internal gains
    try:
        internal_gains = data.get(internal_gains, None)
        internal_gains.set_index(C.DATETIME, inplace=True, drop=False)
        if internal_gains is not None:
            col = obj.get(O.GAINS_INTERNAL_COL)
            col = col if isinstance(col, str) else object_id
            try:
                internal_gains = internal_gains.loc[:, str(col)]
            except KeyError:
                internal_gains = internal_gains.iloc[:, 1]
        else:
            err = f'Internal gains could not be read. Check value.'
            logger.error(err)
            raise ValueError(err)
    except:
        internal_gains = float(obj.get(O.GAINS_INTERNAL, DEFAULT_GAINS_INTERNAL))

    timesteps = weather[C.DATETIME].diff().dt.total_seconds().dropna()
    timestep = timesteps.mode()[0]

    # Precompute relevant data
    solar_gains = calc_solar_gains(active_solar_gains, weather, windows, lat, lon)
    internal_gains = calc_internal_gains(active_internal_gains, weather, internal_gains)

    # Compute temperature and energy demand
    temp_in, p_heat, p_cool = calculate_timeseries(temp_out, temp_init, timestep,
                                                   thermal_resistance, thermal_capacitance, ventilation,
                                                   temp_min, temp_max, active_cool, active_heat,
                                                   power_cool_max, power_heat_max, solar_gains, internal_gains)

    summary = {
        f'{O.DEMAND}_{Types.HEATING}': int(round(p_heat.sum() * timestep / 3600)),
        f'{O.LOAD_MAX}_{Types.HEATING}': int(max(p_heat)),
        f'{O.DEMAND}_{Types.COOLING}': int(round(p_cool.sum() * timestep / 3600)),
        f'{O.LOAD_MAX}_{Types.COOLING}': int(max(p_cool)),
    }
    df = pd.DataFrame({
        C.DATETIME: weather.index,
        f'{C.TEMP_IN}': temp_in.round(2),
        f'{C.LOAD}_{Types.HEATING}': p_heat.astype(int),
        f'{C.LOAD}_{Types.COOLING}': p_cool.astype(int),
    })

    return summary, df


# @njit(cache=True, fastmath=True, parallel=True)
def calculate_timeseries(temp_out, temp_init, timestep, thermal_resistance, thermal_capacitance, ventilation,
                         temp_min, temp_max, active_cool, active_heat, power_cool_max, power_heat_max,
                         solar_gains, internal_gains):
    n_steps = len(temp_out)
    temp_in = np.zeros(n_steps, dtype=np.float64)
    temp_in[0] = temp_init
    p_heat = np.zeros(n_steps, dtype=np.float64)
    p_cool = np.zeros(n_steps, dtype=np.float64)

    # Calculate
    for t in range(1, n_steps):
        # Calculate the net heat transfer
        net_transfer = calc_net_heat_transfer(temp_in[t - 1], temp_out[t], thermal_resistance,
                                              ventilation, solar_gains[t], internal_gains[t])

        # Calculate heating and cooling loads
        p_heat[t] = calc_heating_power(active_heat, net_transfer, temp_in[t - 1], temp_min, thermal_capacitance,
                                       power_heat_max, timestep)
        p_cool[t] = calc_cooling_power(active_cool, net_transfer, temp_in[t - 1], temp_max, thermal_capacitance,
                                       power_cool_max, timestep)

        # Recalculate indoor temperature
        temp_in[t] = calc_temp_in(temp_in[t - 1], net_transfer, p_heat[t], p_cool[t], thermal_capacitance, timestep)

    return temp_in, p_heat, p_cool


def calc_net_heat_transfer(temp_prev, temp_out, thermal_resistance, ventilation, solar_gains, internal_gains):
    """
    Calculate the net passive heat transfer between the indoor space and its environment.
    A positive value indicates a net gain (heat entering) and a negative value indicates a net loss (heat leaving).
    """
    conduction_loss = (temp_out - temp_prev) / thermal_resistance
    ventilation_loss = ventilation * (temp_out - temp_prev)

    return conduction_loss + ventilation_loss + solar_gains + internal_gains


# @njit(cache=True)
def calc_heating_power(active, net_heat_transfer, temp_prev, temp_min, thermal_capacitance, heating_power, timestep):
    """
    Calculate required heating power to bring indoor temperature to temp_min, considering heat gains and losses.
    """
    if not active:
        return 0

    required_heating_power = thermal_capacitance * (temp_min - temp_prev) / timestep - net_heat_transfer

    return min(heating_power, max(0, required_heating_power))


# @njit(cache=True)
def calc_cooling_power(active, net_heat_transfer, temp_prev, temp_max, thermal_capacitance, cooling_power, timestep):
    """
    Calculate required cooling power to bring indoor temperature to temp_max, considering heat gains and losses.
    """
    if not active:
        return 0

    required_cooling_power = thermal_capacitance * (temp_prev - temp_max) / timestep + net_heat_transfer

    return min(cooling_power, max(0, required_cooling_power))


# @njit(cache=True)
def calc_temp_in(temp_in_prev, net_heat_transfer, heating_power, cooling_power, thermal_capacitance, timestep):
    """
    Calculate the new indoor temperature based on energy balance.
    """

    temp_in_new = (temp_in_prev
                   + (timestep / thermal_capacitance)
                   * (net_heat_transfer + heating_power - cooling_power))

    return temp_in_new


def calc_solar_gains(active, weather, windows, lat, lon, model="isotropic"):
    """
    Calculate solar gains for a building.
    """
    if not active or windows is None:
        return np.zeros(len(weather), dtype=np.float32)

    # Obtain all relevant information upfront
    altitude = pvlib.location.lookup_altitude(lat, lon)
    try:
        timezone = weather.index[0].tzinfo
    except Exception:
        timezone = "UTC"
    location = pvlib.location.Location(lat, lon, altitude=altitude, tz=timezone)
    solpos = location.get_solarposition(weather.index)

    # Calculate values depending on model
    if model == 'haydavies':
        dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
        dni = pvlib.irradiance.dirint(ghi=weather[C.SOLAR_GHI],
                                      solar_zenith=solpos['apparent_zenith'],
                                      times=weather.index).fillna(0)
    elif model == 'isotropic':
        dni_extra = None
        dni = weather[C.SOLAR_DNI]
    else:
        logger.error('Unkown irradiance model.')
        raise ValueError('Unknown irradiance model.')

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
            model=model
        )
        poa_global = irr["poa_global"]
        window_gains = poa_global * window["area"] * window["transmittance"] * window["shading"]

        # Accumulate the gains
        total_solar_gains += window_gains.to_numpy(dtype=np.float32)
    return total_solar_gains


def calc_internal_gains(active, weather, gains):
    """
    Calculate internal gains over the time steps.
    If a timeseries is provided, use that.
    Otherwise, if a constant is provided in the object, use that value.
    If neither, return zero internal gains.
    """
    if not active:
        return np.zeros(len(weather), dtype=np.float32)

    # Option 1: Use the provided timeseries
    if isinstance(gains, pd.Series):
        gains = gains.to_numpy(dtype=np.float32)
        if len(gains) != len(weather.index):
            raise ValueError("Length of internal gains timeseries does not match the number of time steps.")
        return gains
    # Option 2: Use a constant provided in the object
    else:
        return np.full(len(weather), gains, dtype=np.float32)
