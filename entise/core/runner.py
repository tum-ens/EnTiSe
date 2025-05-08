import logging
from typing import Dict, Any

from entise.core.registry import get_strategy
from entise.constants import Keys as K

logger = logging.getLogger(__name__)


class RowExecutor:
    def __init__(self, static: Dict[str, Any], timeseries: Dict[str, Any], strategies: Dict[str, str]):
        self.static = static
        self.timeseries = timeseries
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

        logger.debug(f"Running method '{method_cls.__name__}' for type '{ts_type}'")

        result = method.generate(self.static, self.timeseries, ts_type)

        if not isinstance(result, dict) or "timeseries" not in result:
            raise ValueError(f"Invalid result from method '{strategy_name}'")

        self.results[ts_type] = result
        return result

    def run_main_methods(self) -> Dict[str, Dict[str, Any]]:
        for ts_type in self.strategies:
            self.resolve(ts_type)
        return self.results
