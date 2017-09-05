from shapeworld.caption import Predicate, Attribute


class EntityType(Predicate):

    # composed attributes, red and green, square or circle

    __slots__ = ('attributes',)

    def __init__(self, attributes=None):
        if not attributes:
            self.attributes = ()
        elif isinstance(attributes, Attribute):
            self.attributes = (attributes,)
        else:
            assert isinstance(attributes, tuple) or isinstance(attributes, list)
            assert all(isinstance(attribute, Attribute) for attribute in attributes)
            self.attributes = tuple(attributes)

    def model(self):
        return {'component': 'type', 'attributes': [attribute.model() for attribute in self.attributes]}

    def agreeing_entities(self, entities):
        for attribute in self.attributes:
            entities = attribute.agreeing_entities(entities)
        return entities

    def disagreeing_entities(self, entities):
        disagreeing_entities = []
        disagreeing_ids = set()
        for attribute in self.attributes:
            for entity in attribute.disagreeing_entities(entities):
                if entity.id not in disagreeing_ids:
                    disagreeing_entities.append(entity)
                    disagreeing_ids.add(entity.id)
        return disagreeing_entities
