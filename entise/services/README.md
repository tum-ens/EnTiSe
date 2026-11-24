# Services

The [weather service](weather/weather.py) provides weather data for time series
generation. It dispatches requests through the `WeatherProvider` interface.

## Coding constraints

- Keep provider-specific requests and response parsing in the provider, as in
  [`OpenMeteoProvider`](weather/openmeteo.py). Use the shared column constants
  for existing quantities and keep weather names consistent with the
  `name[unit]@height` convention when height applies.
- Mock external provider calls in tests. The Open-Meteo tests patch the client
  and use fixture responses to exercise the provider without a network request.
