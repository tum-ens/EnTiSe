"""The Open-Meteo provider returns start-labelled data on an aware axis."""

from datetime import date, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib
import pytest

from entise.constants import Columns as C
from entise.services.weather import OpenMeteoProvider

LAT, LON, ALT = 49.71754, 11.05877, 300.0
FEATURES = ["temperature_2m", "shortwave_radiation", "precipitation"]
GHI = C.SOLAR_GHI
TEMP = "air_temperature[C]@2m"
RAIN = "precipitation_amount[kg m-2]"


def _fake_client(response_zone: str | None = None):
    """A client that answers like the archive API: hourly labels over the requested local days.

    Each value encodes its label: temperature = 100 + i, radiation and precipitation = i,
    where i counts the hours from the first label.
    """
    calls = []

    def weather_api(url, params):
        calls.append(params)
        zone_name = response_zone if params["timezone"] == "auto" else params["timezone"]
        zone = ZoneInfo(zone_name)
        start = pd.Timestamp(params["start_date"]).tz_localize(zone)
        end = (pd.Timestamp(params["end_date"]) + pd.Timedelta(days=1)).tz_localize(zone)
        labels = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
        i = np.arange(len(labels), dtype=float)
        values = {"temperature_2m": 100.0 + i, "shortwave_radiation": i, "precipitation": i}

        hourly = MagicMock()
        hourly.Time.return_value = int(labels[0].timestamp())
        hourly.TimeEnd.return_value = int(labels[-1].timestamp()) + 3600
        hourly.Interval.return_value = 3600
        hourly.Variables.side_effect = lambda k: MagicMock(
            ValuesAsNumpy=MagicMock(return_value=values[params["hourly"][k]])
        )
        response = MagicMock()
        response.Hourly.return_value = hourly
        response.Latitude.return_value = LAT
        response.Longitude.return_value = LON
        response.Elevation.return_value = ALT
        response.Timezone.return_value = zone_name.encode()
        response.UtcOffsetSeconds.return_value = int(start.utcoffset().total_seconds())
        return [response]

    client = MagicMock()
    client.weather_api.side_effect = weather_api
    return client, calls


def _get(timezone, start="2022-06-01", end="2022-06-01", response_zone=None):
    provider = OpenMeteoProvider()
    provider.client, calls = _fake_client(response_zone)
    df = provider.get_weather_data(
        latitude=LAT, longitude=LON, start_date=start, end_date=end, timezone=timezone, features=FEATURES
    )
    return df, calls


def test_interval_means_move_to_the_start_of_their_interval():
    df, _ = _get("Europe/Berlin")

    # The archive labels an hourly mean at the end of its hour, so the mean over [t, t+1h) is the value at t+1h.
    assert list(df[GHI]) == list(np.arange(1.0, 25.0))
    assert list(df[RAIN]) == list(np.arange(1.0, 25.0))
    # Instantaneous values stay at their label.
    assert list(df[TEMP]) == list(100.0 + np.arange(24.0))


def test_one_extra_day_is_requested_and_trimmed():
    df, calls = _get("Europe/Berlin", start="2022-06-01", end="2022-06-02")

    assert calls[0]["start_date"] == "2022-06-01"
    assert calls[0]["end_date"] == "2022-06-03"
    assert len(df) == 48
    assert df[C.DATETIME].iloc[0] == pd.Timestamp("2022-06-01 00:00", tz="Europe/Berlin")
    assert df[C.DATETIME].iloc[-1] == pd.Timestamp("2022-06-02 23:00", tz="Europe/Berlin")
    assert df[GHI].notna().all()


def test_solar_elevation_is_taken_at_the_interval_midpoint():
    df, _ = _get("Europe/Berlin")

    labels = pd.DatetimeIndex(df[C.DATETIME]).tz_convert("UTC")
    expected = 90 - pvlib.solarposition.get_solarposition(labels + pd.Timedelta(minutes=30), LAT, LON, ALT)["zenith"]
    np.testing.assert_allclose(df["solar_elevation_angle[degree]"], expected.clip(lower=0), atol=1e-6)


def test_without_a_time_zone_the_days_and_labels_are_utc():
    df, calls = _get(None)

    assert calls[0]["timezone"] == "UTC"
    assert str(df[C.DATETIME].dt.tz) == "UTC"
    assert df[C.DATETIME].iloc[0] == pd.Timestamp("2022-06-01 00:00", tz="UTC")
    assert len(df) == 24


def test_a_named_time_zone_is_kept_and_its_daylight_saving_days_have_23_hours():
    df, calls = _get("Europe/Berlin", start="2022-03-27", end="2022-03-27")

    assert calls[0]["timezone"] == "Europe/Berlin"
    assert str(df[C.DATETIME].dt.tz) == "Europe/Berlin"
    assert len(df) == 23


def test_auto_uses_the_zone_the_api_resolved():
    df, calls = _get("auto", response_zone="America/New_York")

    assert calls[0]["timezone"] == "auto"
    assert str(df[C.DATETIME].dt.tz) == "America/New_York"
    assert df[C.DATETIME].iloc[0] == pd.Timestamp("2022-06-01 00:00", tz="America/New_York")
    assert len(df) == 24


def test_the_extra_day_follows_the_calendar():
    _, calls = _get("UTC", start="2022-12-31", end="2022-12-31")

    assert calls[0]["end_date"] == (date(2022, 12, 31) + timedelta(days=1)).isoformat()


@pytest.mark.parametrize("timezone", ["Mars/Olympus", "+01:00"])
def test_an_unknown_time_zone_raises_before_the_request(timezone):
    provider = OpenMeteoProvider()
    provider.client, calls = _fake_client()

    with pytest.raises(ValueError, match="IANA"):
        provider.get_weather_data(
            latitude=LAT, longitude=LON, start_date="2022-06-01", end_date="2022-06-01", timezone=timezone
        )
    assert calls == []
