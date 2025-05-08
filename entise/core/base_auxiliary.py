from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd

class AuxiliaryMethod(ABC):
    name: str
    required_keys: List[str] = []
    optional_keys: List[str] = []
    required_timeseries: List[str] = []
    optional_timeseries: List[str] = []

    def __repr__(self):
        return f"<{self.__class__.__name__} strategy>"

    def __str__(self):
        return self.__class__.__name__

    def generate(self, obj: Dict, data: Dict) -> pd.DataFrame:
        """
        Generate the auxiliary output.

        Returns:
            pd.DataFrame: Resulting time series (e.g., solar gains).
        """
        return self.run(**self.get_input_data(obj, data))

    @abstractmethod
    def get_input_data(self, obj: Dict, data: Dict) -> Dict:
        pass

    @abstractmethod
    def run(self, **kwargs) -> pd.DataFrame:
        pass


class BaseSelector:
    def __init__(self, strategies: List[AuxiliaryMethod]):
        self.strategies = strategies

    def select(self, obj: dict, data: dict) -> AuxiliaryMethod:
        # Sort strategies: most specific (most requirements) first
        sorted_strategies = sorted(
            self.strategies,
            key=lambda s: -(len(getattr(s, "required_keys", [])) + len(getattr(s, "required_timeseries", [])))
        )

        for strategy in sorted_strategies:
            if self._can_apply(strategy, obj, data):
                return strategy
        raise ValueError("No applicable strategy found.")

    @staticmethod
    def _can_apply(strategy: AuxiliaryMethod, obj: dict, data: dict) -> bool:
        # Ensure all required static keys are present
        has_required_keys = all(k in obj and obj[k] is not None for k in strategy.required_keys)

        # Ensure all required timeseries keys are present and not None
        has_required_ts = all(k in data and data[k] is not None for k in strategy.required_timeseries)

        return has_required_keys and has_required_ts



