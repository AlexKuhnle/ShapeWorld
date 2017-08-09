from shapeworld.caption import Predicate, Modifier


class Noun(Predicate):

    # composed modifiers, red and green, square or circle

    __slots__ = ('noun_modifier', 'modifiers')

    def __init__(self, modifiers=None):
        if not modifiers:
            self.modifiers = ()
        elif isinstance(modifiers, Modifier):
            self.modifiers = (modifiers,)
        else:
            assert isinstance(modifiers, tuple) or isinstance(modifiers, list)
            assert all(isinstance(modifier, Modifier) for modifier in modifiers)
            self.modifiers = tuple(modifiers)

    def model(self):
        return {'component': 'noun', 'modifiers': [modifier.model() for modifier in self.modifiers]}

    def agreeing_entities(self, entities):
        for modifier in self.modifiers:
            entities = modifier.agreeing_entities(entities)
        return entities

    def disagreeing_entities(self, entities):
        disagreeing = []
        for modifier in self.modifiers:
            for entity in modifier.disagreeing_entities(entities):
                if entity not in disagreeing:
                    disagreeing.append(entity)
        return disagreeing
