import logging
from typing import Any, Dict

from entise.constants import Keys as K
from entise.core.registry import get_strategy

logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, keys: Dict[str, Any], data: Dict[str, Any], strategies: Dict[str, str]):
        self.keys = keys
        self.data = data
        self.strategies = strategies
        self.results: Dict[str, Dict[str, Any]] = {}

    def resolve(self, ts_type: str) -> Dict[str, Any]:
        if ts_type in self.results:
            return self.results[ts_type]

        strategy_name = self.strategies.get(ts_type)
        if not strategy_name:
            raise ValueError(f"No strategy defined for timeseries type '{ts_type}'")

        method_cls = get_strategy(strategy_name)
        method = method_cls()

        required_ts = getattr(method_cls, K.DATA_REQUIRED, []) or []
        for dep in required_ts:
            if dep in self.strategies:
                # Dependency is produced by a strategy → compute it first
                self.resolve(dep)
            elif dep in self.data:
                # Dependency is provided as input data → accept and continue
                continue

        logger.debug(f"Running method '{method_cls.__name__}' for type '{ts_type}'")

        result = method.generate(self.keys, self.data, self.results, ts_type)

        if not isinstance(result, dict) or K.TIMESERIES not in result:
            raise ValueError(f"Invalid result from method '{strategy_name}'")

        self.results[ts_type] = result
        return result

    def run_methods(self) -> Dict[str, Dict[str, Any]]:
        for ts_type in self.strategies:
            self.resolve(ts_type)
        return self.results
