from __future__ import division
from shapeworld import util
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

    def agreement(self, predication, world):
        rstr_predication = predication.get_sub_predication()
        rstr_body_predication = predication.get_sub_predication()
        assert rstr_body_predication <= rstr_predication

        # if rstr_body_predication.num_agreeing == rstr_body_predication.num_predication:
        #     return 2.0
        # elif rstr_predication.num_not_disagreeing == 0 and body_predication.num_not_disagreeing == 0:
        #     return -2.0
        if rstr_body_predication.num_agreeing > 0:
            return 1.0
        elif rstr_body_predication.num_not_disagreeing == 0:
            return -1.0
        else:
            return 0.0

        # rstr_body_predication = predication.copy()
        # rstr_body_predication.apply(predicate=self.restrictor)
        # rstr_body_predication.apply(predicate=self.body)

        # if rstr_body_predication.num_agreeing == rstr_body_predication.num_predication:
        #     return 2.0
        # elif rstr_predication.num_not_disagreeing == 0 and body_predication.num_not_disagreeing == 0:
        #     return -2.0
        # if rstr_body_predication.num_agreeing > 0:
        #     return 1.0
        # elif rstr_body_predication.num_not_disagreeing == 0:
        #     return -1.0
        # else:
        #     return 0.0
