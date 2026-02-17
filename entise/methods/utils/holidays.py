"""
Holiday helper for demandlib heat (BDEW).

Includes cache for improved performance
"""

import logging
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import holidays

logger = logging.getLogger(__name__)


def _normalize_location(holidays_location: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned = holidays_location.strip().strip('"').strip("'")
    if not cleaned:
        return None, None
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    country = parts[-1].upper() if parts else None
    subdiv = parts[0].upper() if len(parts) > 1 else None
    return country, subdiv


def _years_key(years: Iterable) -> Tuple[int, ...]:
    # Accept ints, strings like "2025", or datetime-like with .year
    ys = set()
    for y in years:
        try:
            y_int = int(getattr(y, "year", y))
        except Exception:
            continue
        ys.add(y_int)
    return tuple(sorted(ys))


@lru_cache(maxsize=256)
def _get_holidays_cached(country: Optional[str], subdiv: Optional[str], years_key: Tuple[int, ...]):
    if not country or not years_key:
        return None
    try:
        cal = holidays.country_holidays(country=country, subdiv=subdiv, years=years_key)
        # Store as tuple to avoid accidental external mutation of cached value
        return tuple(cal.keys())  # tuple[datetime.date, ...]
    except Exception as err:
        logger.warning(
            f"[demandlib heat] Invalid holidays_location or years for country='{country}', subdiv='{subdiv}' → {err}"
        )
        return None


def get_holidays(holidays_location, years):
    if not holidays_location or not isinstance(holidays_location, str):
        return None

    country, subdiv = _normalize_location(holidays_location)
    if not country:
        return None

    ykey = _years_key(years)
    if not ykey:
        return None

    result = _get_holidays_cached(country, subdiv, ykey)
    # Preserve original return type (list) when there is a result
    return list(result) if result is not None else None
