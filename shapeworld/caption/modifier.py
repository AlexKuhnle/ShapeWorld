from shapeworld.caption import Predicate, Settings


class Modifier(Predicate):

    __slots__ = ('modtype', 'value')

    def __init__(self, modtype, value):
        assert modtype in ('relation', 'shape', 'color', 'texture', 'combination', 'shapes', 'colors', 'textures', 'combinations', 'x-max', 'y-max', 'size-max', 'shade-max')
        if modtype == 'relation':
            from shapeworld.caption import Relation
            assert isinstance(value, Relation)
        elif modtype in ('shape', 'color', 'texture', 'combination'):
            assert isinstance(value, str)
        elif modtype in ('shapes', 'colors', 'textures', 'combinations'):
            assert isinstance(value, tuple) or isinstance(value, list)
            value = list(value)
        elif modtype in ('x-max', 'y-max', 'size-max', 'shade-max'):
            assert value == -1 or value == 1
        self.modtype = modtype
        self.value = value

    def model(self):
        if self.modtype == 'relation':
            return dict(component='modifier', modtype=self.modtype, value=self.value.model())
        else:
            return dict(component='modifier', modtype=self.modtype, value=self.value)

    def agreeing_entities(self, entities):
        if self.modtype == 'relation':
            return self.value.agreeing_entities(entities=entities)

        elif self.modtype == 'shape':
            return [entity for entity in entities if entity.shape.name == self.value]

        elif self.modtype == 'color':
            return [entity for entity in entities if entity.color.name == self.value]

        elif self.modtype == 'texture':
            return [entity for entity in entities if entity.texture.name == self.value]

        elif self.modtype == 'combination':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) == self.value]

        elif self.modtype == 'shapes':
            return [entity for entity in entities if entity.shape.name in self.value]

        elif self.modtype == 'colors':
            return [entity for entity in entities if entity.color.name in self.value]

        elif self.modtype == 'textures':
            return [entity for entity in entities if entity.texture.name in self.value]

        elif self.modtype == 'combinations':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) in self.value]

        elif self.modtype == 'x-max':
            # min distance to second
            return [entity for entity in entities if all(entity.center.x * self.value - other.center.x * self.value > Settings.min_distance for other in entities)]

        elif self.modtype == 'y-max':
            # min distance to second
            return [entity for entity in entities if all(entity.center.y * self.value - other.center.y * self.value > Settings.min_distance for other in entities)]

        elif self.modtype == 'size-max':
            # min difference to second
            return [entity for entity in entities if all(entity.shape.area * self.value - other.shape.area * self.value > Settings.min_area for other in entities)]

        elif self.modtype == 'shade-max':
            # min difference to second
            return [entity for entity in entities if all(entity.color.shade * self.value - other.color.shade * self.value > Settings.min_shade for other in entities)]

    def disagreeing_entities(self, entities):
        if self.modtype == 'relation':
            return self.value.disagreeing_entities(entities=entities)

        elif self.modtype == 'shape':
            return [entity for entity in entities if entity.shape.name != self.value]

        elif self.modtype == 'color':
            return [entity for entity in entities if entity.color.name != self.value]

        elif self.modtype == 'texture':
            return [entity for entity in entities if entity.texture.name != self.value]

        elif self.modtype == 'combination':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) != self.value]

        elif self.modtype == 'shapes':
            return [entity for entity in entities if entity.shape.name not in self.value]

        elif self.modtype == 'colors':
            return [entity for entity in entities if entity.color.name not in self.value]

        elif self.modtype == 'textures':
            return [entity for entity in entities if entity.texture.name not in self.value]

        elif self.modtype == 'combinations':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) not in self.value]

        elif self.modtype == 'x-max':
            return [entity for entity in entities if any(other.center.x * self.value - entity.center.x * self.value > Settings.min_distance for other in entities)]

        elif self.modtype == 'y-max':
            return [entity for entity in entities if any(other.center.y * self.value - entity.center.y * self.value > Settings.min_distance for other in entities)]

        elif self.modtype == 'size-max':
            return [entity for entity in entities if any(other.shape.area * self.value - entity.shape.area * self.value > Settings.min_area for other in entities)]

        elif self.modtype == 'shade-max':
            return [entity for entity in entities if any(other.color.shade * self.value - entity.color.shade * self.value > Settings.min_shade for other in entities)]
