import logging
from typing import Any, Dict, Iterable, Tuple

from entise.constants import Keys as K
from entise.core.registry import get_strategy

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, keys: Dict[str, Any], data: Dict[str, Any], strategies: Dict[str, str]):
        self.keys = keys
        self.data = data
        self.strategies = strategies
        self.results: Dict[str, Dict[str, Any]] = {}
        self._resolving: set[str] = set()

    def resolve(self, ts_type: str) -> Dict[str, Any]:
        if ts_type in self.results:
            return self.results[ts_type]

        strategy_name = self.strategies.get(ts_type)
        if not strategy_name:
            raise ValueError(f"No strategy defined for timeseries type '{ts_type}'")
        if ts_type in self._resolving:
            raise ValueError(f"Cyclic strategy dependency for timeseries type '{ts_type}'.")

        method_cls = get_strategy(strategy_name)
        method = method_cls()
        # A method that calculates several of its types at once does so for those that select it.
        generate_group = method.generate_group if ts_type in method.types else None
        group = (ts_type,) if generate_group is None else self._group(method.types, strategy_name)

        self._resolving.update(group)
        try:
            for dep in getattr(method_cls, K.DATA_REQUIRED, []) or []:
                # A dependency produced by a strategy is computed first; one given as input data is used as is.
                if dep in self.strategies:
                    self.resolve(dep)

            logger.debug("Running method '%s' for timeseries types %s", method_cls.__name__, group)
            if generate_group is not None:
                outputs = generate_group(self.keys, self.data, self.results, group)
            else:
                outputs = {ts_type: method.generate(self.keys, self.data, self.results, ts_type)}

            if not isinstance(outputs, dict) or outputs.keys() != set(group):
                raise ValueError(f"Invalid result from method '{strategy_name}': expected results for {group}.")
            for result in outputs.values():
                if not isinstance(result, dict) or K.TIMESERIES not in result:
                    raise ValueError(f"Invalid result from method '{strategy_name}'")
            # Publish the whole group only once every result is valid.
            self.results.update(outputs)
        finally:
            self._resolving.difference_update(group)

        return self.results[ts_type]

    def _group(self, types: Iterable[str], strategy_name: str) -> Tuple[str, ...]:
        """Return the types of ``types`` that select the strategy on this object."""
        name = strategy_name.lower()
        return tuple(kind for kind in types if str(self.strategies.get(kind) or "").lower() == name)

    def run_methods(self) -> Dict[str, Dict[str, Any]]:
        for ts_type in self.strategies:
            self.resolve(ts_type)
        return self.results
