from shapeworld.caption import Predicate


class Noun(Predicate):

    # composed modifiers, red and green, square or circle

    __slots__ = ('predicates',)

    def __init__(self, predicates=None):
        if not predicates:
            self.predicates = ()
        elif isinstance(predicates, Predicate):
            self.predicates = (predicates,)
        else:
            assert isinstance(predicates, tuple) or isinstance(predicates, list)
            assert all(isinstance(predicate, Predicate) for predicate in predicates)
            self.predicates = tuple(predicates)

    def model(self):
        return {'component': 'noun', 'predicates': [predicate.model() for predicate in self.predicates]}

    def agreeing_entities(self, entities):
        for predicate in self.predicates:
            entities = predicate.agreeing_entities(entities)
        return entities

    def disagreeing_entities(self, entities):
        disagreeing = []
        for predicate in self.predicates:
            for entity in predicate.disagreeing_entities(entities):
                if entity not in disagreeing:
                    disagreeing.append(entity)
        return disagreeing
