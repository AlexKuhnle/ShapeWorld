from shapeworld.caption import Clause


class Predicate(Clause):

    def agreeing_entities(self, entities):
        raise NotImplementedError

    def disagreeing_entities(self, entities):
        raise NotImplementedError

    def agreement(self, entities):
        if self.agreeing_entities(entities):
            return 1.0
        elif len(self.disagreeing_entities(entities)) == len(entities):
            return 0.0
        else:
            return 0.5
