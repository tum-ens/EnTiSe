import numpy as np
import pandas as pd
import pvlib

from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.base_auxiliary import AuxiliaryMethod

# Module-level caches (per process)
_SOLPOS_CACHE: dict[tuple, pd.DataFrame] = {}
_POA_CACHE: dict[tuple, np.ndarray] = {}


def _round_loc(lat: float, lon: float, nd: int = 1) -> tuple:
    """Round latitude and longitude to given number of decimals.
    Rounding to 1 decimal gives about 11/7 km precision (lat/lon), enough for caching."""
    return (round(float(lat), nd), round(float(lon), nd))


def _weather_signature_with_ghi(index: pd.DatetimeIndex, ghi: pd.Series) -> tuple:
    """Build a small signature of the weather time grid plus average GHI.

    Returns a tuple (len, first_ts, last_ts, avg_ghi_rounded).
    This keeps keys compact while robustly distinguishing different inputs.
    """
    n = int(len(index))
    first = index[0] if n > 0 else None
    last = index[-1] if n > 0 else None
    # Average GHI rounded for stability; include even if NaN
    try:
        avg_ghi = float(np.nanmean(ghi.to_numpy(dtype=np.float64, copy=False)))
    except Exception:
        avg_ghi = float("nan")
    avg_ghi_r = round(avg_ghi, 3) if np.isfinite(avg_ghi) else avg_ghi
    return (n, first, last, avg_ghi_r)


class SolarGainsInactive(AuxiliaryMethod):
    """
    Handles the creation of solar gains data for cases where solar gains are inactive.

    This class is used to define an auxiliary method that generates a time-series
    of solar gains data with zero values when solar gains calculations are not
    applicable. It ensures compatibility with the required input data and produces
    an output in a specific format.
    """

    required_data = [O.WEATHER]

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
    required_data = [O.WEATHER, O.WINDOWS]

    def get_input_data(self, obj, data):
        object_id = obj[O.ID]
        windows = data.get(O.WINDOWS, None)
        if windows is not None:
            windows = windows.loc[windows[O.ID] == object_id]
            windows = windows if not windows.empty else None
        input_data = {
            "latitude": obj[O.LAT],
            "longitude": obj[O.LON],
            "weather": data[O.WEATHER],
            "windows": windows,
        }
        return input_data

    def run(self, weather, windows, latitude, longitude):
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

        Caching rules:
        - Cache solpos and poa_global.
        - Do NOT cache final total solar gains.
        - Weather identity for caches must depend on location and average GHI in addition to time grid.
        - Window fingerprint for POA cache uses only tilt and orientation.
        """
        if windows is None:
            return pd.DataFrame({O.GAINS_SOLAR: np.zeros(len(weather), dtype=np.float32)}, index=weather.index)

        # Weather/location signatures for caching
        timezone_info = weather.index[0].tzinfo
        tz_offset = 0 if timezone_info is None else timezone_info.utcoffset(None).total_seconds() / 3600
        lat_r, lon_r = _round_loc(latitude, longitude)
        wsig = _weather_signature_with_ghi(weather.index, weather[C.SOLAR_GHI])

        # Cache solar position (solpos)
        sp_key = (wsig, lat_r, lon_r, tz_offset)
        solpos = _SOLPOS_CACHE.get(sp_key)
        if solpos is None:
            location = pvlib.location.Location(latitude, longitude, tz=tz_offset)
            solpos = location.get_solarposition(pd.to_datetime(weather.index, utc=True), method="nrel_numba")
            _SOLPOS_CACHE[sp_key] = solpos

        total_solar_gains = np.zeros(len(weather), dtype=np.float32)
        zenith = solpos["zenith"]
        azimuth = solpos["azimuth"]
        ghi = weather[C.SOLAR_GHI]
        dhi = weather[C.SOLAR_DHI]
        dni = weather[C.SOLAR_DNI]

        # Loop windows; cache POA per (weather/location/tilt/azimuth)
        for _, window in windows.iterrows():
            tilt = float(window[C.TILT])
            orientation = float(window[C.ORIENTATION])
            poa_key = (wsig, lat_r, lon_r, tz_offset, round(tilt, 3), round(orientation, 3))
            poa = _POA_CACHE.get(poa_key)
            if poa is None:
                irr = pvlib.irradiance.get_total_irradiance(
                    surface_tilt=tilt,
                    surface_azimuth=orientation,
                    solar_zenith=zenith,
                    solar_azimuth=azimuth,
                    dni=dni,
                    ghi=ghi,
                    dhi=dhi,
                    dni_extra=None,
                    model="isotropic",
                )
                poa = irr["poa_global"].to_numpy(dtype=np.float32, copy=False)
                _POA_CACHE[poa_key] = poa

            # Compute window gains from POA
            window_gains = poa * float(window[C.AREA]) * float(window[C.G_VALUE]) * float(window[C.SHADING])
            total_solar_gains += window_gains.astype(np.float32, copy=False)

        return pd.DataFrame({O.GAINS_SOLAR: total_solar_gains}, index=weather.index)


class SolarGainsISO13790(SolarGainsPVLib):
    """
    Calculates net solar gains: Solar Irradiance (Gains) - Sky Radiation (Losses).

    This method extends SolarGainsPVLib to include the long-wave radiation heat loss
    to the sky, as required by ISO 13790. It calculates solar gains from both glazed and opaque surfaces,
    and subtracts the sky radiation losses to determine the net solar gains for a building.
    Please note that this implementation uses simplified assumptions for certain parameters
    as per ISO 13790 guidelines.
    """

    # We need envelope properties to calculate the loss area
    required_keys = SolarGainsPVLib.required_keys + [O.H_TR_EM]
    optional_keys = SolarGainsPVLib.optional_keys + [O.H_TR_OP_SKY]

    def get_input_data(self, obj, data):
        # Get standard inputs from parent
        inputs = super().get_input_data(obj, data)

        # Add thermal envelope properties required for sky loss
        # ISO 13790: phi_r = R_se * U * A * h_r * dT_er
        # We use H_tr (U*A) as the proxy for U*A.
        inputs["H_tr_em"] = float(obj.get(O.H_TR_EM))
        h_sky = obj.get(O.H_TR_OP_SKY)

        # If not provided, estimate using 70% factor
        if h_sky is None:
            h_sky = inputs["H_tr_em"] * 0.7  # Default heuristic: Excludes non-sky facing surfaces (e.g., ground)
        inputs["H_tr_op_sky"] = float(h_sky)
        return inputs

    def run(self, weather, windows, latitude, longitude, H_tr_em, H_tr_op_sky):
        # 1. Calculate Glazed Surface Solar Gains (Windows)
        df_gains = super().run(weather, windows, latitude, longitude)
        FRAME_FACTOR = 0.2  # p.70 - 11.4.5: Frame area fraction
        F_W = 0.9  # p.73 - 11.4.2: Non-normal incidence correction
        df_gains = df_gains * F_W * (1.0 - FRAME_FACTOR)  # p.67 - 11.3.3: Glazed gains
        gains_windows = df_gains[O.GAINS_SOLAR].to_numpy()

        # 2. Calculate Opaque Surface Solar Gains (Walls, Roofs)
        # We approximate I_sol with GHI and use H_tr_op_sky as U*A for sky-facing surfaces.
        R_SE = 0.04  # External surface resistance [m2K/W] (simplification)
        F_R = 0.5  # p. Form factor to sky (0.5 = vertical, 1.0 = horizontal) (simplification)
        ALPHA_OP = 0.6  # Solar absorption coefficient (Standard default)
        F_SH_OP = 1.0  # Shading factor
        I_global = weather[C.SOLAR_GHI].to_numpy()  # Using GHI as a proxy for global irradiance on opaque surfaces
        gains_opaque = R_SE * H_tr_op_sky * ALPHA_OP * F_SH_OP * I_global * F_R  # p.68 - 11.3.4: Opaque gains

        # 3. Determine Sky Radiation Losses
        # We approximate (U*A) with H_tr_op_sky
        H_R = 5.0  # p.73 - 11.4.6: External radiative coefficient [W/m2K]
        DT_ER = 11.0  # p.73 - 11.4.6: Average air-sky temperature difference [K]
        loss = F_R * R_SE * H_tr_op_sky * H_R * DT_ER  # p.69 - 11.3.5: Sky radiation losses

        # 4. Net Gains
        net_gains = gains_windows + gains_opaque - loss

        return pd.DataFrame({O.GAINS_SOLAR: net_gains.astype(np.float32)}, index=weather.index)


__all__ = [
    name
    for name, obj in globals().items()
    if isinstance(obj, type)
    and issubclass(obj, AuxiliaryMethod)
    and obj is not AuxiliaryMethod  # exclude the base class
]
