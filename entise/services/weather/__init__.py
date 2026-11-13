from entise._optional import requires_extra

with requires_extra(
    "weather", "openmeteo_requests", "requests_cache", "retry_requests", "pint", "pvlib", feature="Weather service"
):
    from .openmeteo import OpenMeteoProvider
    from .weather import WeatherProvider, WeatherService, logger

__all__ = [
    "WeatherProvider",
    "WeatherService",
    "OpenMeteoProvider",
    "logger",
]
