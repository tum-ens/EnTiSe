from entise.core.base_auxiliary import BaseSelector
from entise.methods.auxiliary.ventilation import strategies

STRATEGIES = [getattr(strategies, name)() for name in strategies.__all__]


class Ventilation(BaseSelector):
    def __init__(self):
        super().__init__(STRATEGIES)

    def generate(self, obj, data):
        strategy = self.select(obj, data)
        return strategy.generate(obj, data)
