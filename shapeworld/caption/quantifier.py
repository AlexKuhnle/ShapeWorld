from __future__ import division
from shapeworld.caption import Clause, EntityType, Relation, Settings


class Quantifier(Clause):

    # can be generalized (no body, able to generate) or not
    # generalized requires entity set

    __slots__ = ('qtype', 'qrange', 'quantity', 'restrictor', 'body')

    def __init__(self, qtype, qrange, quantity, restrictor, body):
        assert qtype in ('absolute', 'relative')
        if qtype == 'absolute':
            assert qrange in ('eq', 'eq-all', 'geq', 'leq', 'neq')
            assert isinstance(quantity, int) and quantity >= 0
        else:
            assert qrange in ('eq', 'geq', 'leq', 'neq')
            assert isinstance(quantity, float) and 0.0 <= quantity <= 1.0
        assert isinstance(restrictor, EntityType)
        assert isinstance(body, Relation)
        self.qtype = qtype
        self.qrange = qrange
        self.quantity = quantity
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return {'component': 'quantifier', 'qtype': self.qtype, 'qrange': self.qrange, 'quantity': self.quantity, 'restrictor': self.restrictor.model(), 'body': self.body.model()}

    def agreement(self, entities):
        num_entities = len(entities)
        restrictor_agreeing_entities = self.restrictor.agreeing_entities(entities=entities)
        num_restrictor_agreeing = len(restrictor_agreeing_entities)
        num_restrictor_disagreeing = len(self.restrictor.disagreeing_entities(entities=entities))
        num_body_disagreeing = len(self.body.disagreeing_entities(entities=entities))
        num_restrictor_body_agreeing = len(self.body.agreeing_entities(entities=restrictor_agreeing_entities))
        num_restrictor_body_disagreeing = len(self.body.disagreeing_entities(entities=restrictor_agreeing_entities))

        if num_restrictor_body_agreeing == num_entities:  # special case: all entities agree
            if self.qrange == 'geq' or (self.qrange in ('eq', 'eq-all') and self.quantity > 0.0):
                return 2.0
            else:
                return -2.0

        elif num_restrictor_disagreeing == num_entities and num_body_disagreeing == num_entities:  # special case: no entities agree
            if self.qrange == 'geq' or (self.qrange in ('eq', 'eq-all') and self.quantity > 0.0):
                return -2.0
            else:
                return 2.0

        if self.qtype == 'absolute':

            if self.qrange == 'eq-all':
                if num_restrictor_agreeing == num_restrictor_body_agreeing == self.quantity:
                    return 1.0
                elif (num_entities - num_restrictor_disagreeing) != self.quantity or (num_entities - num_restrictor_body_disagreeing) != self.quantity:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'eq':
                if num_restrictor_body_agreeing == self.quantity:
                    return 1.0
                elif num_restrictor_agreeing - num_restrictor_body_disagreeing != self.quantity:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'geq':
                if num_restrictor_body_agreeing >= self.quantity:
                    return 1.0
                elif num_restrictor_agreeing - num_restrictor_body_disagreeing < self.quantity:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'leq':
                if num_restrictor_body_agreeing <= self.quantity:
                    return 1.0
                elif num_restrictor_agreeing - num_restrictor_body_disagreeing > self.quantity:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'neq':
                if num_restrictor_body_agreeing != self.quantity:
                    return 1.0
                elif num_restrictor_agreeing - num_restrictor_body_disagreeing == self.quantity:
                    return -1.0
                else:
                    return 0.0

        elif self.qtype == 'relative':

            if num_restrictor_agreeing == 0:  # special case: no agreeing entities for restrictor
                if self.quantity > 0.0:
                    if self.qrange == 'geq' or self.qrange == 'eq':
                        return -1.0
                    else:
                        return 1.0
                else:
                    if self.qrange == 'neq':
                        return -1.0
                    else:
                        return 1.0

            elif self.quantity == 0.0:  # special case: no quantification
                if self.qrange == 'geq':
                    return 1.0
                elif self.qrange == 'neq':
                    if num_restrictor_body_agreeing > 0:
                        return 1.0
                    elif num_restrictor_body_disagreeing == num_restrictor_agreeing:
                        return -1.0
                    else:
                        return 0.0
                else:
                    if num_restrictor_body_agreeing > 0:
                        return -1.0
                    elif num_restrictor_body_disagreeing == num_restrictor_agreeing:
                        return 1.0
                    else:
                        return 0.0

            elif self.quantity == 1.0:  # special case: all quantification
                if self.qrange == 'leq':
                    return 1.0
                elif self.qrange == 'neq':
                    if num_restrictor_body_agreeing == num_restrictor_agreeing:
                        return -1.0
                    elif num_restrictor_body_disagreeing > 0:
                        return 1.0
                    else:
                        return 0.0
                else:
                    if num_restrictor_body_agreeing == num_restrictor_agreeing:
                        return 1.0
                    elif num_restrictor_body_disagreeing > 0:
                        return -1.0
                    else:
                        return 0.0

            elif self.qrange == 'eq':
                if num_restrictor_body_agreeing == int(self.quantity * num_restrictor_agreeing) and int(self.quantity * num_restrictor_agreeing) > 0:  # floor
                    return 1.0
                elif num_restrictor_body_agreeing == int(self.quantity * num_restrictor_agreeing) + bool((self.quantity * num_restrictor_agreeing) % 1):  # ceiling
                    return 1.0
                elif abs(((num_restrictor_agreeing - num_restrictor_body_disagreeing) / num_restrictor_agreeing) - self.quantity) > Settings.min_quantifier:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'geq':
                if num_restrictor_body_agreeing / num_restrictor_agreeing >= self.quantity:
                    return 1.0
                elif self.quantity - ((num_restrictor_agreeing - num_restrictor_body_disagreeing) / num_restrictor_agreeing) > Settings.min_quantifier:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'leq':
                if num_restrictor_body_agreeing / num_restrictor_agreeing <= self.quantity:
                    return 1.0
                elif ((num_restrictor_agreeing - num_restrictor_body_disagreeing) / num_restrictor_agreeing) - self.quantity > Settings.min_quantifier:
                    return -1.0
                else:
                    return 0.0

            elif self.qrange == 'neq':
                if (num_restrictor_agreeing - num_restrictor_body_disagreeing) == int(self.quantity * num_restrictor_agreeing) and int(self.quantity * num_restrictor_agreeing) > 0:  # floor
                    return -1.0
                elif (num_restrictor_agreeing - num_restrictor_body_disagreeing) == int(self.quantity * num_restrictor_agreeing) + bool((self.quantity * num_restrictor_agreeing) % 1):  # ceiling
                    return -1.0
                elif abs((num_restrictor_body_agreeing / num_restrictor_agreeing) - self.quantity) > Settings.min_quantifier:
                    return 1.0
                else:
                    return 0.0
