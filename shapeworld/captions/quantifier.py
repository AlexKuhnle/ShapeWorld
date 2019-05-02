from __future__ import division
from shapeworld.captions import Caption, EntityType, Relation, Settings


class Quantifier(Caption):

    zero_quantifiers = {
        ('count', 'leq', 0), ('count', 'eq', 0), ('count', 'lt', 1),
        ('ratio', 'leq', 0.0), ('ratio', 'eq', 0.0)
    }
    zero_included_quantifiers = {
        ('count', 'lt', '!0'), ('count', 'leq', '*'), ('count', 'eq', 0), ('count', 'neq', '!0'), ('count', 'geq', 0),
        ('ratio', 'lt', '!0'), ('ratio', 'leq', '*'), ('ratio', 'eq', 0.0), ('ratio', 'neq', '!0'), ('ratio', 'geq', 0.0)
    }
    zero_negated_quantifiers = {
        ('count', 'neq', 0), ('count', 'geq', 1), ('count', 'gt', 0),
        ('ratio', 'neq', 0.0), ('ratio', 'gt', 0.0),
    }
    all_quantifiers = {
        ('count', 'geq', -1), ('count', 'eq', -1), ('count', 'gt', -2),
        ('ratio', 'geq', 1.0), ('ratio', 'eq', 1.0)
    }
    all_included_quantifiers = {
        ('count', 'gt', '!1'), ('count', 'geq', '*'), ('count', 'eq', -1), ('count', 'neq', '!1'), ('count', 'leq', -1),
        ('ratio', 'gt', '!1'), ('ratio', 'geq', '*'), ('ratio', 'eq', 1.0), ('ratio', 'neq', '!1'), ('ratio', 'leq', 1.0)
    }
    all_negated_quantifiers = {
        ('count', 'lt', -1), ('count', 'leq', -2), ('count', 'neq', -1),
        ('ratio', 'lt', 1.0), ('ratio', 'neq', 1.0),
    }
    tautological_quantifiers = {
        ('count', 'geq', 0), ('count', 'leq', -1),
        ('ratio', 'geq', 0.0), ('ratio', 'leq', 1.0)
    }

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, body):
        # if qtype == 'composed': qrange is identifier, quantity is list of quantifiers
        assert qtype in ('count', 'ratio', 'composed')
        if qtype == 'count':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, int)
            assert qrange != 'lt' or quantity != 0
            assert qrange != 'gt' or quantity != -1
        elif qtype == 'ratio':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, float) and 0.0 <= quantity <= 1.0
            assert qrange != 'lt' or quantity != 0.0
            assert qrange != 'gt' or quantity != 1.0
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

    def polish_notation(self, reverse=False):
        if reverse:
            if self.qtype == 'composed':
                return self.restrictor.polish_notation(reverse=reverse) + \
                    self.body.polish_notation(reverse=reverse) + \
                    ['{}-{}-{}'.format(self, self.qtype, self.qrange)]
            else:
                return self.restrictor.polish_notation(reverse=reverse) + \
                    self.body.polish_notation(reverse=reverse) + \
                    ['{}-{}-{}-{}'.format(self, self.qtype, self.qrange, self.quantity)]
        else:
            if self.qtype == 'composed':
                return ['{}-{}-{}'.format(self, self.qtype, self.qrange)] + \
                    self.restrictor.polish_notation(reverse=reverse) + \
                    self.body.polish_notation(reverse=reverse)
            else:
                return ['{}-{}-{}-{}'.format(self, self.qtype, self.qrange, self.quantity)] + \
                    self.restrictor.polish_notation(reverse=reverse) + \
                    self.body.polish_notation(reverse=reverse)

    def apply_to_predication(self, predication):
        assert predication.empty()
        rstr_predication = predication.sub_predication()
        self.restrictor.apply_to_predication(predication=rstr_predication)
        body_predication = predication.sub_predication()
        self.body.apply_to_predication(predication=body_predication)
        rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        self.body.apply_to_predication(predication=rstr_body_predication)
        return rstr_predication, body_predication

    def agreement(self, predication, world):
        if self.qtype == 'composed':
            quantifiers = [Quantifier(qtype=quantifier[0], qrange=quantifier[1], quantity=quantifier[2], restrictor=self.restrictor, body=self.body) for quantifier in self.quantity]
            return min(quantifier.agreement(predication=predication.copy(), world=world) for quantifier in quantifiers)

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

        return Quantifier.get_agreement(qrange=self.qrange, lower=lower, upper=upper, target=lower_target, upper_target=upper_target)

    @staticmethod
    def get_agreement(qrange, lower, upper, target, upper_target=None):
        lower_target = target
        if upper_target is None:
            upper_target = target
        assert lower <= upper, (lower, upper)
        assert lower_target <= upper_target, (lower_target, upper_target)

        if lower_target % 1.0 == 0.0 and upper_target % 1.0 == 0.0:
            # no min_quantifier tolerance if quantity is integer
            tolerance = 0.0
        else:
            tolerance = Settings.min_quantifier

        if qrange == 'lt':
            if upper < upper_target - tolerance:
                return 1.0
            elif lower >= lower_target - tolerance:
                return -1.0
            else:
                return 0.0

        elif qrange == 'leq':
            if upper <= upper_target + tolerance:
                return 1.0
            elif lower > lower_target + tolerance:
                return -1.0
            else:
                return 0.0

        elif qrange == 'eq':
            # if lower_target % 1.0 == 0.0:
            #     # special case: no min_quantifier tolerance if quantity is integer
            #     if lower == lower_target and upper == upper_target:
            #         return 1.0
            #     elif lower > lower_target or upper < upper_target:
            #         return -1.0
            #     else:
            #         return 0.0
            if max(upper - upper_target, lower_target - lower) <= tolerance:
                return 1.0
            elif max(lower_target - upper, lower - upper_target) > tolerance:
                return -1.0
            else:
                return 0.0

        elif qrange == 'neq':
            # if lower_target % 1.0 == 0.0:
            #     # special case: no min_quantifier tolerance if quantity is integer
            #     if lower > lower_target or upper < upper_target:
            #         return 1.0
            #     elif lower == lower_target and upper == upper_target:
            #         return -1.0
            #     else:
            #         return 0.0
            if max(lower_target - upper, lower - upper_target) > tolerance:
                return 1.0
            elif max(upper - upper_target, lower_target - lower) <= tolerance:
                return -1.0
            else:
                return 0.0

        elif qrange == 'geq':
            if lower >= lower_target - tolerance:
                return 1.0
            elif upper < upper_target - tolerance:
                return -1.0
            else:
                return 0.0

        elif qrange == 'gt':
            if lower > lower_target + tolerance:
                return 1.0
            elif upper <= upper_target + tolerance:
                return -1.0
            else:
                return 0.0

    @staticmethod
    def tautological(qtype, qrange1, quantity1, qrange2, quantity2):
        if qrange1 == qrange2 and quantity1 == quantity2:
            return True
        elif qrange1 == 'lt':
            if qrange2 in ('lt', 'leq', 'neq') and quantity2 >= quantity1:
                return True
            elif qtype == 'count' and qrange2 == 'leq' and quantity2 == quantity1 - 1:
                return True
        elif qrange1 == 'leq':
            if qrange2 in ('lt', 'leq', 'neq') and quantity2 > quantity1:
                return True
        elif qrange1 == 'eq':
            if qrange2 == 'lt' and quantity2 > quantity1:
                return True
            elif qrange2 == 'leq' and quantity2 >= quantity1:
                return True
            elif qrange2 == 'neq' and quantity2 != quantity1:
                return True
            elif qrange2 == 'geq' and quantity2 <= quantity1:
                return True
            elif qrange2 == 'gt' and quantity2 < quantity1:
                return True
        elif qrange1 == 'neq':
            if qrange2 == 'eq' and quantity2 != quantity1:
                return True
        elif qrange1 == 'geq':
            if qrange2 in ('gt', 'geq', 'neq') and quantity2 < quantity1:
                return True
        elif qrange1 == 'gt':
            if qrange2 in ('gt', 'geq', 'neq') and quantity2 <= quantity1:
                return True
            elif qtype == 'count' and qrange2 == 'geq' and quantity2 == quantity1 + 1:
                return True
        return False

    @staticmethod
    def filter(quantifiers, selection):
        filtered = list()
        for qtype, qrange, quantity in quantifiers:
            for sel_qtype, sel_qrange, sel_quantity in selection:
                if qtype != sel_qtype and sel_qtype != '*':
                    continue
                elif qrange != sel_qrange and sel_qrange != '*':
                    continue
                elif quantity != sel_quantity and sel_quantity != '*' \
                        and (sel_quantity != '!0' or quantity == 0 or quantity == 0.0) \
                        and (sel_quantity != '!1' or quantity == -1 or quantity == 1.0) \
                        and (sel_quantity != '+' or quantity < 0) \
                        and (sel_quantity != '-' or quantity > 0):
                    continue
                else:
                    filtered.append((qtype, qrange, quantity))
        return filtered
