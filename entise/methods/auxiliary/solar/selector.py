from entise.constants import Objects as O
from entise.core.base_auxiliary import BaseSelector
from entise.methods.auxiliary.solar import strategies
from entise.methods.auxiliary.solar.strategies import SolarGainsInactive

STRATEGIES = [getattr(strategies, name)() for name in strategies.__all__]


class SolarGains(BaseSelector):
    def __init__(self):
        super().__init__(STRATEGIES)

    def generate(self, obj, data):
        # Honor the ACTIVE_GAINS_SOLAR flag: when explicitly False, return
        # a zero series instead of computing solar gains. The default is
        # True (active), preserving prior behavior when the flag is absent.
        if not bool(obj.get(O.ACTIVE_GAINS_SOLAR, True)):
            return SolarGainsInactive().generate(obj, data)
        strategy = self.select(obj, data)
        return strategy.generate(obj, data)
