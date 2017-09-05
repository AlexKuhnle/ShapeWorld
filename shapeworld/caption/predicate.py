from shapeworld.caption import Clause


class Predicate(Clause):

    def agreeing_entities(self, entities):
        raise NotImplementedError

    def disagreeing_entities(self, entities):
        raise NotImplementedError

    def agreement(self, entities):
        num_entities = len(entities)
        num_agreeing = len(self.agreeing_entities(entities))
        num_disagreeing = len(self.disagreeing_entities(entities))

        if num_agreeing == num_entities:
            return 2.0
        elif num_agreeing > 0:
            return 1.0
        elif num_disagreeing == num_entities:
            return -1.0
        else:
            return 0.0
