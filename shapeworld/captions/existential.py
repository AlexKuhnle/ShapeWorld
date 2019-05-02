from __future__ import division
from shapeworld.captions import Caption, EntityType, Relation, Selector


class Existential(Caption):

    __slots__ = ('restrictor', 'body')

    def __init__(self, restrictor, body):
        assert isinstance(restrictor, (EntityType, Selector))
        assert isinstance(body, Relation)
        self.restrictor = restrictor
        self.body = body

    def model(self):
        return dict(
            component=str(self),
            restrictor=self.restrictor.model(),
            body=self.body.model()
        )

    def polish_notation(self, reverse=False):
        if reverse:
            return self.restrictor.polish_notation(reverse=reverse) + self.body.polish_notation(reverse=reverse) + [str(self)]
        else:
            return [str(self)] + self.restrictor.polish_notation(reverse=reverse) + self.body.polish_notation(reverse=reverse)

    def apply_to_predication(self, predication):
        assert predication.empty()
        rstr_predication = predication.sub_predication()
        self.restrictor.apply_to_predication(predication=rstr_predication)
        body_predication = predication.sub_predication()
        self.body.apply_to_predication(predication=body_predication)
        # rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        # self.body.apply_to_predication(predication=rstr_body_predication)
        self.body.apply_to_predication(predication=predication)
        self.restrictor.apply_to_predication(predication=predication)
        return rstr_predication, body_predication

    def agreement(self, predication, world):
        rstr_predication = predication.get_sub_predication(0)
        body_predication = predication.get_sub_predication(1)
        # rstr_body_predication = predication.get_sub_predication(2)
        assert predication <= rstr_predication

        if predication.num_agreeing > 0:
            return 1.0
        elif predication.num_not_disagreeing == 0:
            return -1.0
        else:
            return 0.0
