from __future__ import division
from shapeworld.captions import Caption, EntityType, Relation


class Existential(Caption):

    __slots__ = ('restrictor', 'body')

    def __init__(self, restrictor, body):
        assert isinstance(restrictor, EntityType)
        assert isinstance(body, Relation)
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return dict(
            component=str(self),
            restrictor=self.restrictor.model(),
            body=self.body.model()
        )

    def reverse_polish_notation(self):
        return self.restrictor.reverse_polish_notation() + self.body.reverse_polish_notation() + [str(self)]

    def apply_to_predication(self, predication):
        rstr_predication = predication.sub_predication()
        self.restrictor.apply_to_predication(predication=rstr_predication)
        body_predication = predication.sub_predication()
        self.body.apply_to_predication(predication=body_predication)
        rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        self.body.apply_to_predication(predication=rstr_body_predication)
        return rstr_predication, body_predication

    def agreement(self, predication, world):
        rstr_predication = predication.get_sub_predication(0)
        body_predication = predication.get_sub_predication(1)
        rstr_body_predication = predication.get_sub_predication(2)
        assert rstr_body_predication <= rstr_predication

        if rstr_body_predication.num_agreeing > 0:
            return 1.0
        elif rstr_body_predication.num_not_disagreeing == 0:
            return -1.0
        else:
            return 0.0
