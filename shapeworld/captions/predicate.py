from shapeworld import util
from shapeworld.captions import Caption


class Predicate(Caption):

    predtypes = None

    __slots__ = ('predtype', 'value')

    def __init__(self, predtype, value):
        assert self.__class__.predtypes is not None
        assert isinstance(predtype, str) and predtype in self.__class__.predtypes
        self.predtype = predtype
        self.value = value

    def pred_agreement(self, entity, predication):
        raise NotImplementedError

    def pred_disagreement(self, entity, predication):
        raise NotImplementedError

    def filter_agreement(self, entities, predication):
        return [entity for entity in entities if self.pred_agreement(entity=entity, predication=predication)]

    def agreement(self, predication, world):
        # predication.apply(predicate=self)

        # if predication.num_agreeing == predication.num_entities:
        #     return 2.0
        if predication.num_agreeing > 0:
            return 1.0
        elif predication.num_not_disagreeing == 0:
            return -1.0
        else:
            return 0.0
