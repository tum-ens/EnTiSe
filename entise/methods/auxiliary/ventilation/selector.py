from entise.constants import Objects as O
from entise.core.base_auxiliary import BaseSelector
from entise.methods.auxiliary.ventilation import strategies
from entise.methods.auxiliary.ventilation.strategies import VentilationInactive

STRATEGIES = [getattr(strategies, name)() for name in strategies.__all__]


class Ventilation(BaseSelector):
    def __init__(self):
        super().__init__(STRATEGIES)

    def generate(self, obj, data):
        # Honor the ACTIVE_VENTILATION flag: when explicitly False, return
        # a zero series instead of computing ventilation. The default is
        # True (active), preserving prior behavior when the flag is absent.
        if not bool(obj.get(O.ACTIVE_VENTILATION, True)):
            return VentilationInactive().generate(obj, data)
        strategy = self.select(obj, data)
        return strategy.generate(obj, data)
