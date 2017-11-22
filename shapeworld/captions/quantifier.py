from __future__ import division
from shapeworld import util
from shapeworld.captions import Caption, EntityType, Relation, Settings


class Quantifier(Caption):

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, body):
        # if qtype == 'composed': qrange is identifier, quantity is list of quantifiers
        assert qtype in ('count', 'ratio', 'composed')
        if qtype == 'count':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, int)
        elif qtype == 'ratio':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, float) and 0.0 <= quantity <= 1.0
        elif qtype == 'composed':
            assert isinstance(qrange, str)
            assert all(len(quantifier) == 3 and quantifier[0] in ('count', 'ratio') for quantifier in quantity)
            quantity = tuple(tuple(quantifier) for quantifier in quantity)
        assert isinstance(restrictor, EntityType)
        assert isinstance(body, Relation)
        self.qtype = qtype
        self.qrange = qrange
        self.quantity = quantity
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return dict(
            component=str(self),
            qtype=self.qtype,
            qrange=self.qrange,
            quantity=(list(self.quantity) if self.qtype == 'composed' else self.quantity),
            restrictor=self.restrictor.model(),
            body=self.body.model()
        )

    def reverse_polish_notation(self):
        return self.restrictor.reverse_polish_notation() + \
            self.body.reverse_polish_notation() + \
            ['{}-{}-{}-{}'.format(self, self.qtype, self.qrange, self.quantity)]

    def agreement(self, predication, world):
        if self.qtype == 'composed':
            quantifiers = [Quantifier(qtype=quantifier[0], qrange=quantifier[1], quantity=quantifier[2], restrictor=self.restrictor, body=self.body) for quantifier in self.quantity]
            return min(quantifier.agreement(predication=predication.copy(include_sub_predications=True), world=world) for quantifier in quantifiers)

        rstr_predication = predication.get_sub_predication()
        rstr_body_predication = predication.get_sub_predication()
        assert rstr_body_predication <= rstr_predication

        if self.qtype == 'count':
            if self.quantity < 0:
                lower_target = self.quantity + rstr_predication.num_agreeing + 1
                upper_target = self.quantity + rstr_predication.num_not_disagreeing + 1
            else:
                lower_target = self.quantity
                upper_target = None
            lower = rstr_body_predication.num_agreeing
            upper = rstr_body_predication.num_not_disagreeing

        elif self.qtype == 'ratio':
            lower_target = self.quantity
            upper_target = None
            if rstr_predication.num_not_disagreeing == 0:
                lower = 0.0
            else:
                lower = rstr_body_predication.num_agreeing / rstr_predication.num_not_disagreeing
            if rstr_predication.num_agreeing == 0:
                if rstr_body_predication.num_not_disagreeing == 0:
                    upper = 0.0
                else:
                    upper = float('inf')
            else:
                upper = rstr_body_predication.num_not_disagreeing / rstr_predication.num_agreeing

        return Quantifier.get_agreement(qrange=self.qrange, lower=lower, upper=upper, target=lower_target, upper_target=upper_target)

    @staticmethod
    def get_agreement(qrange, lower, upper, target, upper_target=None):
        lower_target = target
        if upper_target is None:
            upper_target = target
        assert lower <= upper
        assert lower_target <= upper_target

        if qrange == 'lt':
            if upper < upper_target:
                return 1.0
            elif lower >= lower_target:
                return -1.0
            else:
                return 0.0

        elif qrange == 'leq':
            if upper <= upper_target:
                return 1.0
            elif lower > lower_target:
                return -1.0
            else:
                return 0.0

        elif qrange == 'eq':
            if lower_target % 1.0 == 0.0:
                # special case: no min_quantifier tolerance if quantity is integer
                if lower == lower_target and upper == upper_target:
                    return 1.0
                elif lower > lower_target or upper < upper_target:
                    return -1.0
                else:
                    return 0.0
            elif max(upper - upper_target, lower_target - lower) < Settings.min_quantifier:
                return 1.0
            elif min(lower - lower_target, upper_target - upper) >= Settings.min_quantifier:
                return -1.0
            else:
                return 0.0

        elif qrange == 'neq':
            if lower_target % 1.0 == 0.0:
                # special case: no min_quantifier tolerance if quantity is integer
                if lower > lower_target or upper < upper_target:
                    return 1.0
                elif lower == lower_target and upper == upper_target:
                    return -1.0
                else:
                    return 0.0
            elif min(lower - lower_target, upper_target - upper) >= Settings.min_quantifier:
                return 1.0
            elif max(upper - upper_target, lower_target - lower) < Settings.min_quantifier:
                return -1.0
            else:
                return 0.0

        elif qrange == 'geq':
            if lower >= lower_target:
                return 1.0
            elif upper < upper_target:
                return -1.0
            else:
                return 0.0

        elif qrange == 'gt':
            if lower > lower_target:
                return 1.0
            elif upper <= upper_target:
                return -1.0
            else:
                return 0.0
