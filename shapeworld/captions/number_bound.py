from __future__ import division
from shapeworld import util
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

    def reverse_polish_notation(self):
        return self.quantifier.reverse_polish_notation() + ['{}-{}'.format(self, self.bound)]

    def agreement(self, predication, world):
        quant_predication = predication.get_sub_predication()
        quant_agreement = self.quantifier.agreement(predication=quant_predication, world=world)
        num_predication = predication.get_sub_predication()

        if num_predication.num_agreeing == self.bound:
            return quant_agreement
        elif num_predication.num_agreeing > self.bound or num_predication.num_not_disagreeing < self.bound:
            return -1.0
        else:
            return min(quant_agreement, 0.0)
