from __future__ import division
from shapeworld.captions import Caption, EntityType, Relation, Settings


class Quantifier(Caption):

    zero_quantifiers = {
        ('count', 'lt', 0), ('count', 'leq', 0), ('count', 'eq', 0), ('count', 'lt', 1),
        ('ratio', 'lt', 0.0), ('ratio', 'leq', 0.0), ('ratio', 'eq', 0.0)
    }
    all_quantifiers = {
        ('count', 'gt', -1), ('count', 'geq', -1), ('count', 'eq', -1), ('count', 'gt', -2),
        ('ratio', 'gt', 1.0), ('ratio', 'geq', 1.0), ('ratio', 'eq', 1.0)
    }
    tautological_quantifiers = {
        ('count', 'geq', 0), ('count', 'leq', -1),
        ('ratio', 'geq', 0.0), ('ratio', 'leq', 1.0)
    }

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, body):
        assert qtype in ('count', 'ratio')
        if qtype == 'count':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, int)
        elif qtype == 'ratio':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, float) and 0.0 <= quantity <= 1.0
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
            quantity=self.quantity,
            restrictor=self.restrictor.model(),
            body=self.body.model()
        )

    def reverse_polish_notation(self):
        return self.restrictor.reverse_polish_notation() + \
            self.body.reverse_polish_notation() + \
            ['{}-{}-{}-{}'.format(self, self.qtype, self.qrange, self.quantity)]

    def apply_to_predication(self, predication):
        rstr_predication = predication.sub_predication()
        self.restrictor.apply_to_predication(predication=rstr_predication)
        body_predication = predication.sub_predication()
        self.body.apply_to_predication(predication=body_predication)
        rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        self.body.apply_to_predication(predication=rstr_body_predication)
        return rstr_predication, body_predication, rstr_body_predication

    def agreement(self, predication, world):
        rstr_predication = predication.get_sub_predication(0)
        body_predication = predication.get_sub_predication(1)
        rstr_body_predication = predication.get_sub_predication(2)
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

        assert lower >= 0.0 and lower_target >= 0.0

        return Quantifier.get_agreement(qrange=self.qrange, lower=lower, upper=upper, target=lower_target, upper_target=upper_target)

    @staticmethod
    def get_agreement(qrange, lower, upper, target, upper_target=None):
        lower_target = target
        if upper_target is None:
            upper_target = target
        assert lower <= upper, (lower, upper)
        assert lower_target <= upper_target, (lower_target, upper_target)

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
            elif max(lower_target - upper, lower - upper_target) >= Settings.min_quantifier:
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
            elif max(lower_target - upper, lower - upper_target) >= Settings.min_quantifier:
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
