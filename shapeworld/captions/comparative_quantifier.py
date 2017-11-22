from __future__ import division
from shapeworld import util
from shapeworld.captions import Caption, EntityType, Relation, Quantifier


class ComparativeQuantifier(Caption):

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'comparison', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, comparison, body):
        # if qtype == 'composed': qrange is identifier, quantity is list of quantifiers
        assert qtype in ('count', 'ratio', 'composed')
        if qtype == 'count':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, int)
        elif qtype == 'ratio':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, float) and quantity >= 0.0
        elif qtype == 'composed':
            assert isinstance(qrange, str)
            assert all(len(quantifier) == 3 and quantifier[0] in ('count', 'ratio') for quantifier in quantity)
            quantity = tuple(tuple(quantifier) for quantifier in quantity)
        assert isinstance(restrictor, EntityType)
        assert isinstance(comparison, EntityType)
        assert isinstance(body, Relation)
        self.qtype = qtype
        self.qrange = qrange
        self.quantity = quantity
        self.restrictor = restrictor
        self.comparison = comparison
        self.body = body

    def model(self):
        return dict(
            component=str(self),
            qtype=self.qtype,
            qrange=self.qrange,
            quantity=(list(self.quantity) if self.qtype == 'composed' else self.quantity),
            restrictor=self.restrictor.model(),
            comparison=self.comparison.model(),
            body=self.body.model()
        )

    def reverse_polish_notation(self):
        return self.restrictor.reverse_polish_notation() + \
            self.comparison.reverse_polish_notation() + \
            self.body.reverse_polish_notation() + \
            ['{}-{}-{}-{}'.format(self, self.qtype, self.qrange, self.quantity)]

    def agreement(self, predication, world):
        if self.qtype == 'composed':
            quantifiers = [Quantifier(qtype=quantifier[0], qrange=quantifier[1], quantity=quantifier[2], restrictor=self.restrictor, comparison=self.comparison, body=self.body) for quantifier in self.quantity]
            return min(quantifier.agreement(predication=predication.copy(include_sub_predications=True), world=world) for quantifier in quantifiers)

        rstr_predication = predication.get_sub_predication()
        rstr_body_predication = predication.get_sub_predication()
        assert rstr_body_predication <= rstr_predication

        comp_predication = predication.get_sub_predication()
        comp_body_predication = predication.get_sub_predication()
        assert comp_body_predication <= comp_predication

        if self.qtype == 'count':
            lower = rstr_body_predication.num_agreeing - comp_body_predication.num_not_disagreeing
            upper = rstr_body_predication.num_not_disagreeing - comp_body_predication.num_agreeing

        elif self.qtype == 'ratio':
            if comp_body_predication.num_not_disagreeing == 0:
                if rstr_body_predication.num_agreeing == 0:
                    lower = 0.0
                else:
                    lower = float('inf')
            else:
                lower = rstr_body_predication.num_agreeing / comp_body_predication.num_not_disagreeing
            if comp_body_predication.num_agreeing == 0:
                if rstr_body_predication.num_not_disagreeing == 0:
                    upper = 0.0
                else:
                    upper = float('inf')
            else:
                upper = rstr_body_predication.num_not_disagreeing / comp_body_predication.num_agreeing

        return Quantifier.get_agreement(qrange=self.qrange, lower=lower, upper=upper, target=self.quantity)
