from __future__ import division
from shapeworld.caption import Clause, Predicate


class Quantifier(Clause):

    # can be generalized (no body, able to generate) or not
    # generalized requires monad-like entity set
    # agreement of 0.5 !!! 0.1 distance for disagreeing etc

    __slots__ = ('qtype', 'qrange', 'quantity', 'tolerance', 'restrictor', 'body')

    def __init__(self, qtype, qrange, quantity, tolerance, restrictor, body):
        assert qtype in ('absolute', 'relative')
        assert qrange in ('eq', 'geq', 'leq', 'neq')
        assert isinstance(quantity, int) == (qtype == 'absolute') and isinstance(quantity, float) == (qtype == 'relative')
        assert quantity >= 0
        assert isinstance(tolerance, float) and 0.0 <= tolerance <= 1.0
        assert isinstance(restrictor, Predicate)
        assert isinstance(body, Predicate)
        self.qtype = qtype
        self.qrange = qrange
        self.quantity = quantity
        self.tolerance = tolerance
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return {'component': 'quantifier', 'qtype': self.qtype, 'qrange': self.qrange, 'quantity': self.quantity, 'tolerance': self.tolerance, 'restrictor': self.restrictor.model(), 'body': self.body.model()}

    def agreement(self, world):
        entities = world['entities']
        restrictor_entities = self.restrictor.agreeing_entities(entities=entities)
        body_entities = self.restrictor.agreeing_entities(entities=self.body.agreeing_entities(entities=entities))

        if self.qtype == 'absolute':
            if self.qrange == 'eq':
                return float(len(body_entities) == self.quantity)
            elif self.qrange == 'geq':
                return float(len(body_entities) >= self.quantity)
            elif self.qrange == 'leq':
                return float(len(body_entities) <= self.quantity)
            elif self.qrange == 'neq':
                return float(len(body_entities) != self.quantity)

        elif self.qtype == 'relative':
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
                elif abs(len(body_entities) / len(restrictor_entities) - self.quantity) > self.tolerance:
                    return 0.0
                else:
                    return 0.5
            elif self.qrange == 'geq':
                if len(body_entities) / len(restrictor_entities) >= self.quantity:
                    return 1.0
                elif self.quantity - (len(body_entities) / len(restrictor_entities)) > self.tolerance:
                    return 0.0
                else:
                    return 0.5
            elif self.qrange == 'leq':
                if len(body_entities) / len(restrictor_entities) <= self.quantity:
                    return 1.0
                elif len(body_entities) / len(restrictor_entities) - self.quantity > self.tolerance:
                    return 0.0
                else:
                    return 0.5
            elif self.qrange == 'neq':
                if len(body_entities) / len(restrictor_entities) == self.quantity:
                    return 0.0
                elif abs(len(body_entities) / len(restrictor_entities) - self.quantity) > self.tolerance:
                    return 1.0
                else:
                    return 0.5
