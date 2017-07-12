from shapeworld.world import Shape
from shapeworld.caption import Predicate


class Modifier(Predicate):

    __slots__ = ('modtype', 'value')

    def __init__(self, modtype, value=None):
        assert modtype in ('relation', 'shape', 'color', 'texture', 'combination', 'shapes', 'colors', 'textures', 'combinations', 'x-max', 'y-max', 'size-max', 'shade-max')
        self.modtype = modtype
        if modtype == 'relation':
            from shapeworld.caption import Relation
            assert isinstance(value, Relation)
            self.value = value
        elif modtype in ('shape', 'color', 'texture', 'combination'):
            assert isinstance(value, str)
            self.value = value
        elif modtype in ('shapes', 'colors', 'textures', 'combinations'):
            assert isinstance(value, tuple) or isinstance(value, list)
            self.value = tuple(value)
        elif modtype in ('x-max', 'y-max', 'shade-max'):
            assert value == -1 or value == 1
            self.value = value
            # assert (isinstance(value, tuple) or isinstance(value, list)) and len(value) == 2
            # self.value = Point(value)

    def model(self):
        if self.modtype in ('shapes', 'colors', 'textures', 'combinations', 'location-max'):
            value = list(self.value)
        else:
            value = self.value
        return dict(component='modifier', modtype=self.modtype, value=value)

    def agreeing_entities(self, entities):
        if self.modtype == 'relation':
            return self.value.agreeing_entities(entities=entities)

        elif self.modtype == 'shape':
            return [entity for entity in entities if entity['shape']['name'] == self.value]

        elif self.modtype == 'color':
            return [entity for entity in entities if entity['color']['name'] == self.value]

        elif self.modtype == 'texture':
            return [entity for entity in entities if entity['texture']['name'] == self.value]

        elif self.modtype == 'combination':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) == self.value]

        elif self.modtype == 'shapes':
            return [entity for entity in entities if entity['shape']['name'] in self.value]

        elif self.modtype == 'colors':
            return [entity for entity in entities if entity['color']['name'] in self.value]

        elif self.modtype == 'textures':
            return [entity for entity in entities if entity['texture']['name'] in self.value]

        elif self.modtype == 'combinations':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) in self.value]

        elif self.modtype == 'x-max':
            # not really a modifier!
            # min distance to second
            return [entity for entity in entities if all(entity['center']['x'] * self.value > other['center']['x'] * self.value for other in entities)]

        elif self.modtype == 'y-max':
            # min distance to second
            return [entity for entity in entities if all(entity['center']['y'] * self.value > other['center']['y'] * self.value for other in entities)]

        elif self.modtype == 'size-max':
            # min difference to second
            entity_areas = [Shape.from_model(entity).area() for entity in entities]
            return [entity for entity, area in zip(entities, entity_areas) if all(area * self.value > other * self.value for other in entity_areas)]

        elif self.modtype == 'shade-max':
            # min difference to second
            return [entity for entity in entities if all(entity['color']['shade'] * self.value > other['color']['shade'] * self.value for other in entities)]

    def disagreeing_entities(self, entities):
        if self.modtype == 'relation':
            return self.value.disagreeing_entities(entities=entities)

        elif self.modtype == 'shape':
            return [entity for entity in entities if entity['shape']['name'] != self.value]

        elif self.modtype == 'color':
            return [entity for entity in entities if entity['color']['name'] != self.value]

        elif self.modtype == 'texture':
            return [entity for entity in entities if entity['texture']['name'] != self.value]

        elif self.modtype == 'combination':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) != self.value]

        elif self.modtype == 'shapes':
            return [entity for entity in entities if entity['shape']['name'] not in self.value]

        elif self.modtype == 'colors':
            return [entity for entity in entities if entity['color']['name'] not in self.value]

        elif self.modtype == 'textures':
            return [entity for entity in entities if entity['texture']['name'] not in self.value]

        elif self.modtype == 'combinations':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) not in self.value]

        elif self.modtype == 'x-max':
            return [entity for entity in entities if any(entity['center']['x'] * self.value < other['center']['x'] * self.value for other in entities)]

        elif self.modtype == 'y-max':
            return [entity for entity in entities if any(entity['center']['y'] * self.value < other['center']['y'] * self.value for other in entities)]

        elif self.modtype == 'size-max':
            entity_areas = [Shape.from_model(entity).area() for entity in entities]
            return [entity for entity, area in zip(entities, entity_areas) if any(area * self.value < other * self.value for other in entity_areas)]

        elif self.modtype == 'shade-max':
            return [entity for entity in entities if any(entity['color']['shade'] * self.value < other['color']['shade'] * self.value for other in entities)]
