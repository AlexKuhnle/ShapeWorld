from __future__ import division
from shapeworld.captions import Caption, Quantifier


class NumberBound(Caption):

    __slots__ = ('bound', 'quantifier')

    def __init__(self, bound, quantifier):
        assert isinstance(bound, int) and bound >= 0
        assert isinstance(quantifier, Quantifier)
        self.bound = bound
        self.quantifier = quantifier

    def model(self):
        return dict(
            component=str(self),
            bound=self.bound,
            quantifier=self.quantifier.model()
        )

    def polish_notation(self, reverse=False):
        if reverse:
            return self.quantifier.polish_notation(reverse=reverse) + ['{}-{}'.format(self, self.bound)]
        else:
            return ['{}-{}'.format(self, self.bound)] + self.quantifier.polish_notation(reverse=reverse)

    def apply_to_predication(self, predication):
        assert predication.empty()
        quant_predication = predication.sub_predication()
        self.quantifier.apply_to_predication(predication=quant_predication)
        num_predication = predication.sub_predication()
        self.quantifier.restrictor.apply_to_predication(predication=num_predication)
        return num_predication

    def agreement(self, predication, world):
        quant_predication = predication.get_sub_predication(0)
        quant_agreement = self.quantifier.agreement(predication=quant_predication, world=world)
        num_predication = predication.get_sub_predication(1)

        if num_predication.num_agreeing == self.bound:
            return quant_agreement
        elif num_predication.num_agreeing > self.bound or num_predication.num_not_disagreeing < self.bound:
            return -1.0
        else:
            return min(quant_agreement, 0.0)
