from entise.constants import Objects as O
from entise.core.base_auxiliary import BaseSelector
from entise.methods.auxiliary.internal import strategies
from entise.methods.auxiliary.internal.strategies import InternalInactive

STRATEGIES = [getattr(strategies, name)() for name in strategies.__all__]


class InternalGains(BaseSelector):
    def __init__(self):
        super().__init__(STRATEGIES)

    def generate(self, obj, data):
        # Honor the ACTIVE_GAINS_INTERNAL flag: when explicitly False, return
        # a zero series instead of computing internal gains. The default is
        # True (active), preserving prior behavior when the flag is absent.
        if not bool(obj.get(O.ACTIVE_GAINS_INTERNAL, True)):
            return InternalInactive().generate(obj, data)
        strategy = self.select(obj, data)
        return strategy.generate(obj, data)
