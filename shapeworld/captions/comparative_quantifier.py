from __future__ import division
from shapeworld.captions import Caption, EntityType, Relation, Quantifier


class ComparativeQuantifier(Caption):

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'comparison', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, comparison, body):
        assert qtype in ('count', 'ratio')
        if qtype == 'count':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, int)
        elif qtype == 'ratio':
            assert qrange in ('lt', 'leq', 'eq', 'neq', 'geq', 'gt')
            assert isinstance(quantity, float) and quantity > 0.0
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
            quantity=self.quantity,
            restrictor=self.restrictor.model(),
            comparison=self.comparison.model(),
            body=self.body.model()
        )

    def reverse_polish_notation(self):
        return self.restrictor.reverse_polish_notation() + \
            self.comparison.reverse_polish_notation() + \
            self.body.reverse_polish_notation() + \
            ['{}-{}-{}-{}'.format(self, self.qtype, self.qrange, self.quantity)]

    def apply_to_predication(self, predication):
        rstr_predication = predication.sub_predication()
        self.restrictor.apply_to_predication(predication=rstr_predication)
        rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        self.body.apply_to_predication(predication=rstr_body_predication)

        comp_predication = predication.sub_predication()
        self.comparison.apply_to_predication(predication=comp_predication)
        comp_body_predication = predication.sub_predication(predication=comp_predication.copy())
        self.body.apply_to_predication(predication=comp_body_predication)

        body_predication = predication.sub_predication()
        self.body.apply_to_predication(predication=body_predication)

        return rstr_predication, rstr_body_predication, comp_predication, comp_body_predication, body_predication

    def agreement(self, predication, world):
        rstr_predication = predication.get_sub_predication(0)
        rstr_body_predication = predication.get_sub_predication(1)
        assert rstr_body_predication <= rstr_predication

        comp_predication = predication.get_sub_predication(2)
        comp_body_predication = predication.get_sub_predication(3)
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
