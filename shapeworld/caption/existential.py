from __future__ import division
from shapeworld.caption import Clause, EntityType, Relation


class Existential(Clause):

    __slots__ = ('restrictor', 'body')

    def __init__(self, restrictor, body):
        assert isinstance(restrictor, EntityType)
        assert isinstance(body, Relation)
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return {'component': 'existential', 'restrictor': self.restrictor.model(), 'body': self.body.model()}

    def agreement(self, entities):
        body_agreeing_entities = self.body.agreeing_entities(entities=entities)
        num_entities = len(entities)
        num_restrictor_disagreeing = len(self.restrictor.disagreeing_entities(entities=entities))
        num_body_agreeing = len(body_agreeing_entities)
        num_body_disagreeing = len(self.body.disagreeing_entities(entities=entities))
        num_restrictor_body_agreeing = len(self.restrictor.agreeing_entities(entities=body_agreeing_entities))
        num_restrictor_body_disagreeing = len(self.restrictor.disagreeing_entities(entities=body_agreeing_entities))
        # print(num_entities, num_restrictor_disagreeing, num_body_agreeing, num_body_disagreeing, num_restrictor_body_agreeing, num_restrictor_body_disagreeing)

        if num_restrictor_body_agreeing == num_entities:
            return 2.0
        elif num_restrictor_body_agreeing > 0:
            return 1.0
        elif num_restrictor_disagreeing == num_entities and num_body_disagreeing == num_entities:
            return -2.0
        elif num_restrictor_body_disagreeing == num_body_agreeing:
            return -1.0
        else:
            return 0.0
