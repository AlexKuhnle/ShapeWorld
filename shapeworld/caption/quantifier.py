from __future__ import division
from shapeworld.caption import Clause, Noun, Relation, Settings


class Quantifier(Clause):

    # can be generalized (no body, able to generate) or not
    # generalized requires entity set
    # agreement of 0.5 !!! 0.1 distance for disagreeing etc

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, body):
        assert qtype in ('absolute', 'relative')
        if qtype == 'absolute':
            assert qrange in ('eq', 'eq-all', 'geq', 'leq', 'neq')
            assert isinstance(quantity, int) and quantity >= 0
        else:
            assert qrange in ('eq', 'geq', 'leq', 'neq')
            assert isinstance(quantity, float) and 0.0 <= quantity <= 1.0
        assert isinstance(restrictor, Noun)
        assert isinstance(body, Relation)
        self.qtype = qtype
        self.qrange = qrange
        self.quantity = quantity
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return {'component': 'quantifier', 'qtype': self.qtype, 'qrange': self.qrange, 'quantity': self.quantity, 'restrictor': self.restrictor.model(), 'body': self.body.model()}

    def agreement(self, entities):
        body_entities = self.restrictor.agreeing_entities(entities=self.body.agreeing_entities(entities=entities))

        if self.qtype == 'absolute':
            if self.qrange == 'eq':
                return float(len(body_entities) == self.quantity)
            if self.qrange == 'eq-all':
                restrictor_entities = self.restrictor.agreeing_entities(entities=entities)
                return float(len(body_entities) == len(restrictor_entities) == self.quantity)
            elif self.qrange == 'geq':
                return float(len(body_entities) >= self.quantity)
            elif self.qrange == 'leq':
                return float(len(body_entities) <= self.quantity)
            elif self.qrange == 'neq':
                return float(len(body_entities) != self.quantity)

        elif self.qtype == 'relative':
            restrictor_entities = self.restrictor.agreeing_entities(entities=entities)

            if len(restrictor_entities) == 0:  # special case: no agreeing entities
                if self.qrange == 'neq':
                    return float(self.quantity != 0.0)
                elif self.qrange == 'leq':
                    return 1.0
                else:
                    return float(self.quantity == 0.0)
            elif self.quantity == 0.0:  # special case: no quantification
                if self.qrange == 'geq':
                    return 1.0
                elif self.qrange == 'neq':
                    return float(len(body_entities) != 0)
                else:
                    return float(len(body_entities) == 0)
            elif self.quantity == 1.0:  # special case: all quantification
                if self.qrange == 'leq':
                    return 1.0
                elif self.qrange == 'neq':
                    return float(len(body_entities) != len(restrictor_entities))
                else:
                    return float(len(body_entities) == len(restrictor_entities))

            elif self.qrange == 'eq':
                if len(body_entities) / len(restrictor_entities) == self.quantity:
                    return 1.0
                elif abs(len(body_entities) / len(restrictor_entities) - self.quantity) > Settings.min_quantifier:
                    return 0.0
                else:
                    return 0.5
            elif self.qrange == 'geq':
                if len(body_entities) / len(restrictor_entities) >= self.quantity:
                    return 1.0
                elif self.quantity - (len(body_entities) / len(restrictor_entities)) > Settings.min_quantifier:
                    return 0.0
                else:
                    return 0.5
            elif self.qrange == 'leq':
                if len(body_entities) / len(restrictor_entities) <= self.quantity:
                    return 1.0
                elif len(body_entities) / len(restrictor_entities) - self.quantity > Settings.min_quantifier:
                    return 0.0
                else:
                    return 0.5
            elif self.qrange == 'neq':
                if len(body_entities) == int(self.quantity * len(restrictor_entities)):  # floor
                    return 0.0
                elif len(body_entities) == int(self.quantity * len(restrictor_entities)) + bool((self.quantity * len(restrictor_entities)) % 1):  # ceiling
                    return 0.0
                elif abs(len(body_entities) / len(restrictor_entities) - self.quantity) > Settings.min_quantifier:
                    return 1.0
                else:
                    return 0.5
