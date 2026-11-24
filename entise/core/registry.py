import importlib
import pkgutil
from typing import Collection, List, NamedTuple, Optional, Type

from entise.constants import VALID_TYPES, Types
from entise.core.base import Method, method_registry


class _BuiltinStrategy(NamedTuple):
    module: str
    types: Collection[str]
    extra: Optional[str]


# Module names are relative to entise.methods. Keep the established discovery order.
_BUILTIN_STRATEGIES = {
    "internalgainsoccupancy": _BuiltinStrategy("auxiliary.internal.strategies", (), None),
    "jordanvajen": _BuiltinStrategy("dhw.jordan_vajen", (Types.DHW,), "dhw-jordanvajen"),
    "demandlib_electricity": _BuiltinStrategy("electricity.demandlib", (Types.ELECTRICITY,), "electricity-demandlib"),
    "pylpg": _BuiltinStrategy("electricity.pylpg", (Types.ELECTRICITY,), "electricity-pylpg"),
    "demandlib_heat": _BuiltinStrategy("heat.demandlib", (Types.HEATING,), "heating-demandlib"),
    "districtheatingsim": _BuiltinStrategy("heat.districtheatingsim", (Types.HEATING,), None),
    "ruhnau": _BuiltinStrategy("hp.ruhnau", (Types.HP,), "hp-ruhnau"),
    "1r1c": _BuiltinStrategy("hvac.R1C1", (Types.HVAC, Types.HEATING, Types.COOLING), "hvac-1r1c"),
    "5r1c": _BuiltinStrategy("hvac.R5C1", (Types.HVAC, Types.HEATING, Types.COOLING), "hvac-5r1c"),
    "1r0c": _BuiltinStrategy("hvac.R1C0", (Types.HVAC, Types.HEATING, Types.COOLING), "hvac-1r0c"),
    "7r2c": _BuiltinStrategy("hvac.R7C2", (Types.HVAC, Types.HEATING, Types.COOLING), "hvac-7r2c"),
    "file": _BuiltinStrategy("multiple.file", VALID_TYPES, "file"),
    "geoma": _BuiltinStrategy("occupancy.geoma", (Types.OCCUPANCY,), "occupancy-geoma"),
    "pht": _BuiltinStrategy("occupancy.pht", (Types.OCCUPANCY,), "occupancy-pht"),
    "pvlib": _BuiltinStrategy("pv.pvlib", (Types.PV,), "pv-pvlib"),
    "wplib": _BuiltinStrategy("wind.wplib", (Types.WIND,), "wind-wplib"),
}


def register(cls: Type[Method], strategy_name: str = None):
    name = (strategy_name or getattr(cls, "name", cls.__name__)).lower()
    if name in method_registry or name in _BUILTIN_STRATEGIES:
        raise ValueError(f"Duplicate strategy name: '{name}' is already registered.")
    method_registry[name] = cls


def get_strategy(strategy_name: str) -> Type[Method]:
    name = strategy_name.lower()
    strategy = _BUILTIN_STRATEGIES.get(name)
    registered = method_registry.get(name)
    if strategy is not None:
        module_name = f"entise.methods.{strategy.module}"
        if registered is None or registered.__module__ == module_name:
            # Wait for module initialization if another thread is importing it.
            importlib.import_module(module_name)
    if name not in method_registry:
        raise ValueError(f"Strategy '{strategy_name}' not found.")
    return method_registry[name]


def get_strategy_extra(strategy_name: str) -> Optional[str]:
    """Return the install extra for a built-in strategy, if it has one."""
    strategy = _BUILTIN_STRATEGIES.get(strategy_name.lower())
    return strategy.extra if strategy else None


def list_strategies() -> list:
    """List built-in and registered custom strategy names without loading methods."""
    return list(dict.fromkeys((*_BUILTIN_STRATEGIES, *method_registry)))


def get_methods_by_type(ts_type: str):
    """Load and return classes for one time-series type."""
    classes = []
    for name in list_strategies():
        if name in method_registry:
            types = getattr(method_registry[name], "types", ())
        else:
            types = _BUILTIN_STRATEGIES[name].types
        if ts_type in types:
            classes.append(get_strategy(name))
    return classes


def import_all_methods(path: Optional[List[str]] = None, package_name: Optional[str] = None):
    """Explicitly import all public method modules from a package."""
    if path is None or package_name is None:
        raise ValueError("Both path and package_name must be provided to avoid circular imports.")

    for _, modname, _ in pkgutil.walk_packages(path, package_name + "."):
        if any(part.startswith("_") for part in modname.split(".")):
            continue
        importlib.import_module(modname)
