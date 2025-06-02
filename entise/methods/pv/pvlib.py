import logging
import warnings

import pandas as pd
import pvlib
from pvlib import pvsystem

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Types
import entise.methods.utils as utils

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

    def generate(self, obj, data, ts_type: str = Types.PV):

        obj, data = get_input_data(obj, data)

        ts = calculate_timeseries(obj, data)

        logger.debug(f"[PV pvlib]: Generating {ts_type} data")

        timestep = data[O.WEATHER][C.DATETIME].diff().dt.total_seconds().dropna().mode()[0]
        summary = {
            f'{C.GENERATION}_{Types.PV}': (ts.sum() * timestep / 3600).round().astype(int),
            f'{O.GEN_MAX}_{Types.PV}': ts.max().round().astype(int),
            f'{C.FLH}_{Types.PV}': (ts.sum() * timestep / 3600 / obj[O.POWER]).round().astype(int),
        }

        ts = ts.rename(columns={'p_mp': f'{C.POWER}_{Types.PV}'})

        return {
            "summary": summary,
            "timeseries": ts,
        }

def get_input_data(obj, data):
    obj_out = {
        O.ID: utils.get_with_backup(obj, O.ID),
        O.ALTITUDE: utils.get_with_backup(obj, O.ALTITUDE),
        O.AZIMUTH: utils.get_with_backup(obj, O.AZIMUTH, AZIMUTH),
        O.LAT: utils.get_with_backup(obj, O.LAT),
        O.LON: utils.get_with_backup(obj, O.LON),
        O.POWER: utils.get_with_backup(obj, O.POWER, POWER),
        O.PV_ARRAYS: utils.get_with_backup(obj, O.PV_ARRAYS),
        O.PV_INVERTER: utils.get_with_backup(obj, O.PV_INVERTER),
        O.TILT: utils.get_with_backup(obj, O.TILT, TILT),
    }
    data_out = {
        O.WEATHER: utils.get_with_backup(data, O.WEATHER),
        O.PV_ARRAYS: utils.get_with_backup(data, obj_out[O.PV_ARRAYS], PV_ARRAYS),
        O.PV_INVERTER: utils.get_with_backup(data, O.PV_INVERTER, PV_INVERTER),
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
