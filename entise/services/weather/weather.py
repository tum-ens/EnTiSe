"""Weather data service module.

This module provides a modular approach to retrieving weather data from various sources.
It includes a base WeatherProvider interface, specific provider implementations,
and a WeatherService facade for easy access to weather data.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import pint
import pvlib

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


class WeatherProvider(ABC):
    """Abstract base class for all weather data providers."""

    @abstractmethod
    def get_weather_data(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        timezone: Optional[str] = None,
        features: Optional[List[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieve weather data for a specific location and time period."""
        raise NotImplementedError

    def convert_units(self, weather: pd.DataFrame, metadata: Dict[str, dict]) -> pd.DataFrame:
        """Convert weather data columns to CF standard units based on metadata."""
        for col in weather.columns:
            meta = metadata.get(col)
            if not meta:
                continue
            if meta.get("unit") != meta.get("cf_unit"):
                weather[col] = self._convert_unit_column(weather[col], meta["unit"], meta["cf_unit"])
        return weather

    @staticmethod
    def use_default_features(metadata: Dict[str, dict]) -> set:
        """Return set of default features based on metadata dictionary."""
        return [key for key, val in metadata.items() if val.get("default") is True]

    @staticmethod
    def _validate_features(features: List[str], available_features: Dict[str, dict]) -> List[str]:
        """Ensure requested features are in the list of available features."""
        missing = [f for f in features if f not in available_features]
        if missing:
            raise ValueError(
                f"Requested features not available: {missing}. " f"Available: {sorted(available_features)}"
            )
        return features

    def calculate_derived_variables(self, df: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate derived weather variables like absolute humidity."""
        result = df.copy()
        if variables is None or "absolute_humidity" in variables:
            if all(col in result.columns for col in ["relative_humidity_2m", "temperature_2m", "surface_pressure"]):
                result["absolute_humidity"] = self._compute_absolute_humidity(
                    result["relative_humidity_2m"], result["temperature_2m"], result["surface_pressure"]
                )
        return result

    @staticmethod
    def _compute_absolute_humidity(
        relative_humidity: pd.Series, temperature: pd.Series, pressure: pd.Series
    ) -> pd.Series:
        """Compute absolute humidity [kg/m³] from relative humidity [%], temperature [°C], and pressure [hPa]."""
        P_sat = 6.112 * np.exp((17.67 * temperature) / (temperature + 237.3))  # hPa
        P_v = relative_humidity / 100 * P_sat  # hPa
        absolute_humidity = (0.622 * P_v) / (pressure - P_v)  # kg/kg dry air
        return absolute_humidity

    @staticmethod
    def _calculate_sun_elevation(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        freq: pd.Timedelta,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> pd.Series:
        """Calculate sun elevation angle [°] for a given location and time range."""
        times = pd.date_range(start=start_date, end=end_date, freq=freq, inclusive="left")
        solar_pos = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)
        elevation = 90 - solar_pos["zenith"]
        return elevation.clip(lower=0)

    @staticmethod
    def _parse_height_suffix(name: str) -> Tuple[str, Optional[str]]:
        """Extract variable base name and height suffix (converted to meters if needed)."""
        match = re.match(r"^(.+?)_(\d+)(c?)m$", name)
        if match:
            base, val, is_cm = match.groups()
            suffix = f"_{float(val) / 100:.2f}m" if is_cm else f"_{val}m"
            return base, suffix
        return name, None

    @staticmethod
    def _convert_to_cf_name(openmeteo_name: str, cf_map: Dict[str, dict] | Dict[str, str]) -> str:
        """Convert Open-Meteo variable name to CF standard name with height suffix (if any).

        Supports two mapping styles:
        - Dict[str, dict]: provider FEATURES-like mapping where value holds a 'cf' key.
        - Dict[str, str]: simple base->cf mapping.
        """
        # Extract a simple height suffix like _2m or _10m or _12cm; complex ranges are ignored here
        base, suffix = WeatherProvider._parse_height_suffix(openmeteo_name)
        cf_base: str | None = None
        # If a FEATURES-like mapping is passed, prefer direct lookup on full name first
        if isinstance(cf_map.get(openmeteo_name, None), dict):
            meta = cf_map.get(openmeteo_name, {})  # type: ignore[index]
            cf_base = meta.get("cf") if isinstance(meta, dict) else None
        # Fallback: try mapping with the parsed base for simple dicts
        if cf_base is None:
            # When cf_map is Dict[str, str], map the parsed base; otherwise try dict-of-dict again
            mapped = cf_map.get(base)  # type: ignore[call-arg]
            if isinstance(mapped, dict):
                cf_base = mapped.get("cf")
            elif isinstance(mapped, str):
                cf_base = mapped
        if cf_base is None:
            # Last resort: if we can find any dict on the full name, use its 'cf', else keep the original base
            maybe = cf_map.get(openmeteo_name)  # type: ignore[call-arg]
            if isinstance(maybe, dict):
                cf_base = maybe.get("cf", base)
            else:
                cf_base = base
        return f"{cf_base}{suffix or ''}"

    @staticmethod
    def _extract_height(openmeteo_name: str) -> Optional[str]:
        """Extract measurement height as a string in meters from common patterns.

        Examples:
            temperature_2m -> "2m"
            wind_speed_100m -> "100m"
            soil_temperature_7cm -> "0.07m"
        Complex layer ranges like "_0_to_7cm" are ignored (return None).
        """
        m = re.search(r"_(\d+)m$", openmeteo_name)
        if m:
            return f"{m.group(1)}m"
        cm = re.search(r"_(\d+)cm$", openmeteo_name)
        if cm:
            val = float(cm.group(1)) / 100.0
            return f"{val}m"
        return None

    @staticmethod
    def _convert_to_enriched_name(openmeteo_name: str, metadata: Dict[str, dict]) -> str:
        """Build enriched CF column name including units and optional height.

        Pattern:
            <cf_name>[<cf_unit>] or <cf_name>[<cf_unit>]@<height>
        """
        meta = metadata.get(openmeteo_name, {})
        cf_name = meta.get("cf") if isinstance(meta, dict) else None
        if not cf_name:
            # Fallback to CF conversion (may append _<height> suffix we ignore for enriched)
            cf_name = WeatherProvider._convert_to_cf_name(openmeteo_name, metadata)
            # Strip trailing _<number>m from cf_name to keep base for enriched
            cf_name = re.sub(r"_(?:\d+)(?:\.\d+)?m$", "", cf_name)
        cf_unit = meta.get("cf_unit") if isinstance(meta, dict) else None
        enriched = cf_name if not cf_unit else f"{cf_name}[{cf_unit}]"
        height = WeatherProvider._extract_height(openmeteo_name)
        if height:
            enriched = f"{enriched}@{height}"
        return enriched

    @staticmethod
    def convert_with_pint(values: pd.Series, src_unit: str, tgt_unit: str) -> pd.Series:
        try:
            quantity = Q_(values.values, src_unit)
            converted = quantity.to(tgt_unit).magnitude
            return pd.Series(converted, index=values.index)
        except pint.errors.DimensionalityError:
            raise ValueError(f"Incompatible units: {src_unit} → {tgt_unit}")
        except pint.errors.UndefinedUnitError:
            raise ValueError(f"Unknown unit: {src_unit} or {tgt_unit}")

    def _convert_unit_column(self, values: pd.Series, src_unit: str, tgt_unit: str) -> pd.Series:
        if src_unit == "%" and tgt_unit == "1":
            return values / 100
        if src_unit == "mm" and tgt_unit == "kg m-2":
            return values  # assume 1 mm = 1 kg/m²
        # fallback to pint
        return self.convert_with_pint(values, src_unit, tgt_unit)


class WeatherService:
    """Facade for accessing weather data from different providers."""

    def __init__(self):
        """Initialize the weather service."""
        from entise.services.weather.openmeteo import OpenMeteoProvider

        self.providers = {"openmeteo": OpenMeteoProvider()}

    def register_provider(self, name: str, provider: WeatherProvider):
        """Register a new weather data provider.

        Args:
            name: Name of the provider
            provider: Provider instance
        """
        if not isinstance(provider, WeatherProvider):
            raise TypeError("Provider must implement WeatherProvider interface")
        self.providers[name] = provider

    def get_weather_data(self, provider: Literal["openmeteo"] = "openmeteo", **kwargs) -> pd.DataFrame:
        """Get weather data using the specified provider.

        Args:
            provider: Name of the provider to use
            **kwargs: Provider-specific parameters

        Returns:
            DataFrame with weather data
        """
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not found")

        return self.providers[provider].get_weather_data(**kwargs)

    # The following helper and facade methods mirror those used in provider tests
    def calculate_derived_variables(self, df: pd.DataFrame, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate derived weather variables like absolute humidity.

        This mirrors the provider-level calculation but is exposed here for convenience
        and to support existing tests that call it on the service instance.
        """
        result = df.copy()
        if variables is None or "absolute_humidity" in variables:
            needed = ["relative_humidity_2m", "temperature_2m", "surface_pressure"]
            if all(col in result.columns for col in needed):
                result["absolute_humidity"] = self._calculate_absolute_humidity(
                    result["relative_humidity_2m"], result["temperature_2m"], result["surface_pressure"]
                )
        return result

    @staticmethod
    def _calculate_absolute_humidity(
        relative_humidity: pd.Series, temperature: pd.Series, pressure: pd.Series
    ) -> pd.Series:
        """Compute absolute humidity [kg/m³] from relative humidity [%], temperature [°C], and pressure [hPa]."""
        P_sat = 6.112 * np.exp((17.67 * temperature) / (temperature + 237.3))  # hPa
        P_v = relative_humidity / 100 * P_sat  # hPa
        absolute_humidity = (0.622 * P_v) / (pressure - P_v)  # kg/kg dry air
        return absolute_humidity


if __name__ == "__main__":
    # Example usage
    weather_service = WeatherService()

    # Get weather data for a specific location and time period
    df = weather_service.get_weather_data(
        latitude=49.71754, longitude=11.05877, start_date="2021-12-31", end_date="2023-01-01", timezone="Europe/Berlin"
    )

    # Save to CSV
    df.to_csv("weather_check2.csv", index=False)

    print(f"Retrieved weather data with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
