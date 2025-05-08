import numpy as np
import pandas as pd
import logging

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types
from entise.methods.auxiliary.solar.selector import SolarGains
from entise.methods.auxiliary.internal.selector import InternalGains

logger = logging.getLogger(__name__)

# Default values for optional keys
DEFAULT_ACTIVE_COOLING = True
DEFAULT_ACTIVE_HEATING = True
DEFAULT_ACTIVE_SOLAR_GAINS = True
DEFAULT_ACTIVE_INTERNAL_GAINS = True
DEFAULT_POWER_HEATING = np.inf
DEFAULT_POWER_COOLING = np.inf
DEFAULT_TEMP_INIT = 22
DEFAULT_TEMP_MAX = 24
DEFAULT_TEMP_MIN = 20
DEFAULT_THERMAL_INERTIA = 0
DEFAULT_VENTILATION = 0


class R1C1(Method):
    types = [Types.HVAC]
    name = "1R1C"
    required_keys = [O.CAPACITANCE, O.RESISTANCE, O.WEATHER]
    optional_keys = [O.POWER_HEATING, O.POWER_COOLING, O.ACTIVE_HEATING, O.ACTIVE_COOLING, O.VENTILATION, O.TEMP_INIT, O.TEMP_MIN, O.TEMP_MAX]
    required_timeseries = [O.WEATHER]
    optional_timeseries = [O.GAINS_INTERNAL, O.GAINS_SOLAR]
    output_summary = {
            f'{C.DEMAND}_{Types.HEATING}': 'total heating demand',
            f'{C.DEMAND}_{Types.COOLING}': 'total cooling demand',
    }
    output_timeseries = {
            f'{C.TEMP_IN}': 'indoor temperature',
            f'{C.LOAD}_{Types.HEATING}': 'heating load',
            f'{C.LOAD}_{Types.COOLING}': 'cooling load',
    }

    def generate(self, obj, data, ts_type):

        obj, data = get_input_data(obj, data)

        # Precompute auxiliary data
        data[O.GAINS_SOLAR] = SolarGains().generate(obj, data)
        data[O.GAINS_INTERNAL] = InternalGains().generate(obj, data)

        # Compute temperature and energy demand
        temp_in, p_heat, p_cool = calculate_timeseries(obj, data)

        logger.debug(f"[HVAC R1C1] {ts_type}: max heating {p_heat.max()}, cooling {p_cool.max()}")

        timestep = data[O.WEATHER][C.DATETIME].diff().dt.total_seconds().dropna().mode()[0]
        summary = {
            f"{C.DEMAND}_{Types.HEATING}": int(round(p_heat.sum() * timestep / 3600)),
            f'{O.LOAD_MAX}_{Types.HEATING}': int(max(p_heat)),
            f"{C.DEMAND}_{Types.COOLING}": int(round(p_cool.sum() * timestep / 3600)),
            f'{O.LOAD_MAX}_{Types.COOLING}': int(max(p_cool)),
        }

        df = pd.DataFrame({
            f"{C.TEMP_IN}": temp_in,
            f"{C.LOAD}_{Types.HEATING}": p_heat,
            f"{C.LOAD}_{Types.COOLING}": p_cool,
        }, index= data[O.WEATHER].index)
        df.index.name = C.DATETIME

        return {
            "summary": summary,
            "timeseries": df
        }

def get_input_data(obj: dict, data: dict) -> tuple[dict, dict]:
    obj_out = {
        O.ID: obj.get(O.ID, None),
        O.ACTIVE_COOLING: obj.get(O.ACTIVE_COOLING, DEFAULT_ACTIVE_COOLING),
        O.ACTIVE_HEATING: obj.get(O.ACTIVE_HEATING, DEFAULT_ACTIVE_HEATING),
        O.GAINS_INTERNAL: obj.get(O.GAINS_INTERNAL, None),
        O.GAINS_INTERNAL_COL: obj.get(O.GAINS_INTERNAL_COL, None),
        O.GAINS_SOLAR: obj.get(O.GAINS_SOLAR, None),
        O.LAT: obj.get(O.LAT, None),
        O.LON: obj.get(O.LON, None),
        O.POWER_COOLING: obj.get(O.POWER_COOLING, DEFAULT_POWER_COOLING),
        O.POWER_HEATING: obj.get(O.POWER_HEATING, DEFAULT_POWER_HEATING),
        O.RESISTANCE: obj[O.RESISTANCE],
        O.CAPACITANCE: obj[O.CAPACITANCE],
        O.TEMP_INIT: obj.get(O.TEMP_INIT, DEFAULT_TEMP_INIT),
        O.TEMP_MAX: obj.get(O.TEMP_MAX, DEFAULT_TEMP_MAX),
        O.TEMP_MIN: obj.get(O.TEMP_MIN, DEFAULT_TEMP_MIN),
        O.VENTILATION: obj.get(O.VENTILATION, DEFAULT_VENTILATION),
    }
    internal_key = obj.get(O.GAINS_INTERNAL)
    internal_gains = data.get(internal_key) if isinstance(internal_key, str) else None
    data_out = {
        O.WEATHER: data.get(O.WEATHER, None),
        O.WINDOWS: data.get(O.WINDOWS, None),
        internal_key: internal_gains,
    }

    # Clean up
    obj_out = {k: v for k, v in obj_out.items() if v is not None}
    data_out = {k: v for k, v in data_out.items() if v is not None}

    # Safe datetime handling
    if O.WEATHER in data_out:
        weather = data_out[O.WEATHER].copy()
        weather[C.DATETIME] = pd.to_datetime(weather[C.DATETIME])
        weather.set_index(C.DATETIME, inplace=True, drop=False)
        data_out[O.WEATHER] = weather

    return obj_out, data_out


def calculate_timeseries(obj: dict, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Get objects
    temp_init = obj[O.TEMP_INIT]
    thermal_resistance = obj[O.RESISTANCE]
    thermal_capacitance = obj[O.CAPACITANCE]
    ventilation = obj[O.VENTILATION]
    temp_min = obj[O.TEMP_MIN]
    temp_max = obj[O.TEMP_MAX]
    active_cool = obj[O.ACTIVE_COOLING]
    active_heat = obj[O.ACTIVE_HEATING]
    power_cool_max = obj[O.POWER_COOLING]
    power_heat_max = obj[O.POWER_HEATING]
    # Get data
    solar_gains = data[O.GAINS_SOLAR].to_numpy(dtype=np.float32)
    internal_gains = data[O.GAINS_INTERNAL].to_numpy(dtype=np.float32)
    weather = data[O.WEATHER]
    temp_out = weather[C.TEMP_OUT].to_numpy(dtype=np.float32) if C.TEMP_OUT in weather else None
    if temp_out is None:
        raise Exception(f"Missing temperature column: {C.TEMP_OUT}")

    timesteps = weather[C.DATETIME].diff().dt.total_seconds().dropna()
    timestep = timesteps.mode()[0]

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
        temp_in[t] = calc_temp_in(temp_in[t - 1], net_transfer, p_heat[t], p_cool[t], thermal_capacitance,
                                  timestep)

    return temp_in, p_heat, p_cool


def calc_net_heat_transfer(temp_prev, temp_out, thermal_resistance, ventilation, solar_gains, internal_gains):
    """
    Calculate the net passive heat transfer between the indoor space and its environment.
    A positive value indicates a net gain (heat entering) and a negative value indicates a net loss (heat leaving).
    """
    conduction_loss = (temp_out - temp_prev) / thermal_resistance
    ventilation_loss = ventilation * (temp_out - temp_prev)

    return conduction_loss + ventilation_loss + solar_gains + internal_gains


def calc_heating_power(active, net_heat_transfer, temp_prev, temp_min, thermal_capacitance, heating_power,
                       timestep):
    """
    Calculate required heating power to bring indoor temperature to temp_min, considering heat gains and losses.
    """
    if not active:
        return 0

    required_heating_power = thermal_capacitance * (temp_min - temp_prev) / timestep - net_heat_transfer

    return _ensure_scalar(min(heating_power, max(0, required_heating_power)))


def calc_cooling_power(active, net_heat_transfer, temp_prev, temp_max, thermal_capacitance, cooling_power,
                       timestep):
    """
    Calculate required cooling power to bring indoor temperature to temp_max, considering heat gains and losses.
    """
    if not active:
        return 0

    required_cooling_power = thermal_capacitance * (temp_prev - temp_max) / timestep + net_heat_transfer

    return _ensure_scalar(min(cooling_power, max(0, required_cooling_power)))


def calc_temp_in(temp_in_prev, net_heat_transfer, heating_power, cooling_power, thermal_capacitance, timestep):
    """
    Calculate the new indoor temperature based on energy balance.
    """

    temp_in_new = (temp_in_prev
                   + (timestep / thermal_capacitance)
                   * (net_heat_transfer + heating_power - cooling_power))

    return _ensure_scalar(temp_in_new)

def _ensure_scalar(x):
    return x.item() if isinstance(x, np.ndarray) else x