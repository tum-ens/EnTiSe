"""Typed heating and cooling results from one solve of an RC model (#127).

The RC models calculate their types together through ``GroupedMethod``. This module
adds the rules that only the RC models have:

- the control of each selected type keeps the object's setting, which can be a
  schedule, and the control of a type the model is not selected for is off;
- a type-scoped key may not set an input of the other control;
- ``hvac`` and a typed key cannot select the same model on one object.

Each typed result holds the indoor temperature and the columns and summary keys
prefixed with its type.
"""

import logging
from typing import Any, Iterable, Tuple

import numpy as np

from entise.constants import SEP, Types
from entise.constants import Columns as C
from entise.constants import Objects as O
from entise.core.grouped import GroupedMethod

logger = logging.getLogger(__name__)

TYPED = (Types.HEATING, Types.COOLING)

# The inputs that belong to the heating or cooling control. All other inputs describe the building.
CONTROL_KEYS = {
    Types.HEATING: (O.ACTIVE_HEATING, O.POWER_HEATING, O.TEMP_MIN),
    Types.COOLING: (O.ACTIVE_COOLING, O.POWER_COOLING, O.TEMP_MAX, O.TARGET_HUMIDITY_REL),
}
SWITCHES = {Types.HEATING: O.ACTIVE_HEATING, Types.COOLING: O.ACTIVE_COOLING}


class TypedViews(GroupedMethod):
    """Mixin for an RC model ``Method`` whose ``types`` are ``[hvac, heating, cooling]``.

    A direct call of ``generate`` with ``ts_type`` set to ``heating`` or ``cooling``
    goes through ``group_result``.
    """

    shared_columns = (C.TEMP_IN,)
    control_keys = CONTROL_KEYS

    def generate_group(self, obj: dict, data: dict, results: dict | None, ts_types: Iterable[str]) -> dict:
        """Return the result of each type in ``ts_types`` from one solve."""
        ts_types = tuple(ts_types)
        if ts_types == (Types.HVAC,):
            return {Types.HVAC: self.generate(obj, data, results, Types.HVAC)}
        if Types.HVAC in ts_types:
            typed = [kind for kind in ts_types if kind != Types.HVAC]
            raise ValueError(
                f"Object '{obj.get(O.ID)}': '{Types.HVAC}' and {', '.join(repr(kind) for kind in typed)} "
                f"all select {self.name}. Use either '{Types.HVAC}' for the full result or "
                f"'{Types.HEATING}' and '{Types.COOLING}' for the typed results."
            )
        return super().generate_group(obj, data, results, ts_types)

    def generate_combined(self, obj: dict, data: dict, results: dict | None, ts_types: Tuple[str, ...]) -> dict:
        """Solve the building once; the result is the full ``hvac`` result."""
        return self.generate(obj, data, results, Types.HVAC)

    def group_object(self, obj: dict, ts_types: Tuple[str, ...]) -> dict:
        """Resolve the inputs as ``GroupedMethod`` does, then set the controls of the selected types."""
        self._check_scoped_keys(obj, ts_types)
        flat = super().group_object(obj, ts_types)
        for kind in TYPED:
            switch = SWITCHES[kind]
            if kind not in ts_types:
                flat[switch] = False
            elif _is_off(flat.get(switch)):
                logger.warning(
                    "Object '%s' selects %s for '%s', but '%s' is off, so its %s load is zero.",
                    obj.get(O.ID),
                    self.name,
                    kind,
                    switch,
                    kind,
                )
        return flat

    def _check_scoped_keys(self, obj: dict, ts_types: Tuple[str, ...]) -> None:
        declared = self.key_defaults()
        for kind in ts_types:
            prefix = f"{kind}{SEP}"
            other = [t for t in TYPED if t != kind]
            for key in obj:
                if not (isinstance(key, str) and key.startswith(prefix)):
                    continue
                base = key.removeprefix(prefix)
                for owner in other:
                    if base in declared and base in self.control_keys.get(owner, ()):
                        raise ValueError(
                            f"Object '{obj.get(O.ID)}': '{key}' sets an input of the {owner} control of "
                            f"{self.name}. Give it as '{base}' or '{owner}{SEP}{base}'."
                        )


def _is_off(value: Any) -> bool:
    """Whether a control setting is a constant 'off'; schedules and table references are not."""
    return isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)) and not value
