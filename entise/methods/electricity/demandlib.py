import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from entise._optional import require_module, requires_extra
from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core._time import output_index, site_zone
from entise.core.base import Method
from entise.methods.utils._calendar import read_profile, scale_to_energy, time_table, wall_years
from entise.methods.utils._demandlib import contain_warning_filters
from entise.methods.utils.holidays import get_holidays

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "h0_dyn"
DEFAULT_DEMAND = 1.0


class Demandlib(Method):
    """Electricity demand using demandlib's BDEW standard load profiles (SLPs).

    This method wraps demandlib's implementation of the German BDEW standard load
    profiles to generate high‑quality electricity demand time series. Given a
    target time horizon (``Objects.DATETIMES``) and an energy demand in kWh over
    that horizon, it constructs the canonical 15‑minute SLP for the relevant
    calendar year(s) and reads it at the site's local wall clock.

    Key characteristics:
    - Profile families such as household (H0) and commercial (Gx) are supported
      via ``Objects.PROFILE``; defaults to an H0 dynamic profile.
    - Public holidays can be considered by providing ``Objects.HOLIDAYS_LOCATION``.
    - Output power is provided in Watts. Each row is the mean over the
      interval that starts at its label; the energy over the run equals
      ``Objects.DEMAND_KWH``.
    - The local calendar comes from the object's ``timezone`` (UTC if unset).
      On the 23-hour day of the change to summer time the skipped hour is not
      read; on the 25-hour day the repeated hour is read twice.

    Reference:
    - demandlib documentation (BDEW SLPs): https://demandlib.readthedocs.io/
    - BDEW guideline (German Association of Energy and Water Industries).
    """

    name = "demandlib_electricity"
    types = [Types.ELECTRICITY]

    required_keys = [O.DATETIMES]
    optional_keys = {
        O.DEMAND_KWH: DEFAULT_DEMAND,
        O.PROFILE: DEFAULT_PROFILE,
        O.HOLIDAYS_LOCATION: None,
        O.TIMEZONE: None,
    }

    required_data = [O.DATETIMES]

    output_summary = {
        f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]": "total electricity demand",
        f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]": "maximum electricity load",
    }
    output_timeseries = {
        f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]": "electricity load",
    }

    input_aliases = {"demand_kwh": O.DEMAND_KWH}

    def generate(
        self,
        obj: dict | None = None,
        data: dict | None = None,
        results: dict | None = None,
        ts_type: str = Types.ELECTRICITY,
        **inputs,
    ) -> dict:
        """Generate an electricity demand timeseries using demandlib BDEW profiles.

        This is the public entry point for the electricity method. It accepts either
        an object/data mapping or keyword overrides, prepares inputs, computes the
        load profile at the requested resolution, and returns a summary and
        timeseries dataframe.

        Args:
            obj: Object dictionary providing inputs (e.g., demand, profile key).
            data: Data dictionary containing required timeseries (O.DATETIMES).
            results: Unused placeholder for interface compatibility.
            ts_type: Timeseries type, defaults to Types.ELECTRICITY.
            **inputs: Any input of the method, by its key or by its Python name (the key
                without its unit, such as ``area`` for ``area[m2]``). Keywords override the
                same keys in ``obj``. A table input, such as ``datetimes``, can be a
                DataFrame. For example:

                - ``profile`` (str): BDEW profile key (e.g., "h0", "g0"); defaults to
                  DEFAULT_PROFILE.
                - ``demand`` or ``demand_kwh`` (``demand[kWh]``): Demand in kWh over the run; defaults to 1.0.
                - ``datetimes`` (DataFrame or str): The run's datetimes, or the key of their
                  table in ``data``.
                - ``holidays_location`` (str): Holidays country/region code used by demandlib to adjust
                  profiles (e.g., "DE").

        Returns:
            dict: A dictionary with keys:
                - "summary": Mapping with total demand [Wh] and max load [W].
                - "timeseries": DataFrame indexed by the run's instants in the site time zone, with one
                  column "ELECTRICITY|load[W]" of integer W values.
        """
        processed_obj, processed_data = self._process_kwargs(obj, data, ts_type, **inputs)
        processed_obj, processed_data = self._get_input_data(processed_obj, processed_data, ts_type)

        ts = calculate_timeseries(processed_obj, processed_data)

        logger.debug("[demandlib elec]: Generated succesfully.")

        return self._format_output(ts, processed_obj, processed_data)

    def _get_input_data(self, obj: dict, data: dict, method_type: str = Types.ELECTRICITY) -> tuple[dict, dict]:
        obj_out = self.get_relevant_objects(obj, method_type)

        if float(obj_out[O.DEMAND_KWH]) <= 0:
            raise ValueError("[demandlib_electricity] demand must be > 0 ")

        dts_key = Method.get_with_method_backup(obj, O.DATETIMES, method_type, O.DATETIMES)
        table, step, hours = time_table(data, dts_key, site_zone(obj_out))
        data_out = {O.DATETIMES: table, "step": step, "hours": hours, "run_data": data}
        return obj_out, data_out

    @staticmethod
    def _format_output(ts: pd.DataFrame, obj: dict, data: dict) -> dict:
        out_col = f"{Types.ELECTRICITY}{SEP}{C.LOAD}[W]"

        ts.columns = [out_col]

        demand_wh = int(round(float(np.dot(ts[out_col].to_numpy(dtype=float), data["hours"]))))
        load_max_w = int(round(float(ts[out_col].max())))

        summary = {
            f"{Types.ELECTRICITY}{SEP}{C.DEMAND}[Wh]": demand_wh,
            f"{Types.ELECTRICITY}{SEP}{O.LOAD_MAX}[W]": load_max_w,
        }

        ts.index = output_index(ts.index, obj, data["run_data"])

        return {"summary": summary, "timeseries": ts}


def calculate_timeseries(processed_obj: Dict[str, Any], processed_data: Dict[str, Any]) -> pd.DataFrame:
    """Compute the electricity load timeseries at the requested resolution.

    The BDEW standard load profile is built for every calendar year that the
    run's local wall clock touches. Each row of the run, which covers
    ``[t, t + step)``, takes the mean of the profile over the same span of the
    site's legal wall clock (see :mod:`entise.methods.utils._calendar`). On the
    23-hour day the skipped hour is not read, and on the 25-hour day the
    repeated hour is read twice. The result is scaled so that the energy over
    the run equals the demand.

    Args:
        processed_obj: Validated object mapping (contains O.PROFILE, O.DEMAND_KWH,
            and optional O.HOLIDAYS_LOCATION and O.TIMEZONE).
        processed_data: Mapping with the run's normalized O.DATETIMES table and its step.

    Returns:
        pandas.DataFrame: Single-column DataFrame of integer Watts indexed by
        the run's UTC instants.
    """
    profile_type = processed_obj[O.PROFILE].lower()
    demand_kwh = float(processed_obj[O.DEMAND_KWH])
    holidays_location = processed_obj.get(O.HOLIDAYS_LOCATION)
    zone = site_zone(processed_obj)

    axis = processed_data[O.DATETIMES].index
    step = processed_data["step"]

    years = wall_years(axis, zone, step)
    with requires_extra("electricity-demandlib", "demandlib"):
        from demandlib import bdew
    require_module("electricity-demandlib", "holidays")
    holidays = get_holidays(holidays_location=holidays_location, years=years)

    # The profile's shape on the local calendar; the scale is set by the demand over the run.
    with contain_warning_filters():
        profile = pd.concat(
            [
                bdew.ElecSlp(y, holidays=holidays).get_scaled_power_profiles({profile_type: 1.0})[profile_type]
                for y in years
            ]
        )

    power = read_profile(profile, axis, zone, step, name="demandlib_electricity")
    power = scale_to_energy(power, processed_data["hours"], demand_kwh * 1000.0)

    return pd.DataFrame({profile_type: np.round(power).astype(int)}, index=axis)
