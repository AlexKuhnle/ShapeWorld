from shapeworld.caption import Clause


class Predicate(Clause):

    def agreeing_entities(self, entities):
        raise NotImplementedError

    def disagreeing_entities(self, entities):
        raise NotImplementedError

    def agreement(self, world):
        if self.agreeing_entities(world['entities']):
            return 1.0
        elif len(self.disagreeing_entities(world['entities'])) == len(world['entities']):
            return 0.0
        else:
            return 0.5
