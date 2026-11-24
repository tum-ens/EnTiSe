# Sample weather

[Weather data by Open-Meteo](https://open-meteo.com/) in `weather.csv`
is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The file contains hourly data for 2022. EnTiSe renamed fields and added unit
and height labels, converted relative humidity from percent to a fraction
and surface pressure from hPa to Pa, and stored timestamps with UTC offsets.
Hourly irradiance values were relabelled from the end to the start of each
interval. EnTiSe calculated and added the mid-interval solar elevation angle
and renamed Open-Meteo's terrestrial radiation column to
`toa_incoming_shortwave_flux[W m-2]`.

See the [Open-Meteo data license](https://open-meteo.com/en/licence)
and [hosted API terms](https://open-meteo.com/en/terms).
