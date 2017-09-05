from shapeworld.caption import Predicate, Settings


class Attribute(Predicate):

    __slots__ = ('attrtype', 'value')

    def __init__(self, attrtype, value):
        assert attrtype in ('relation', 'shape', 'color', 'texture', 'combination', 'shapes', 'colors', 'textures', 'combinations', 'x-max', 'y-max', 'size-max', 'shade-max')
        if attrtype == 'relation':
            from shapeworld.caption import Relation
            assert isinstance(value, Relation)
        elif attrtype in ('shape', 'color', 'texture', 'combination'):
            assert isinstance(value, str)
        elif attrtype in ('shapes', 'colors', 'textures', 'combinations'):
            assert isinstance(value, tuple) or isinstance(value, list)
            value = list(value)
        elif attrtype in ('x-max', 'y-max', 'size-max', 'shade-max'):
            assert value == -1 or value == 1
        self.attrtype = attrtype
        self.value = value

    def model(self):
        if self.attrtype == 'relation':
            return dict(component='attribute', attrtype=self.attrtype, value=self.value.model())
        else:
            return dict(component='attribute', attrtype=self.attrtype, value=self.value)

    def agreeing_entities(self, entities):
        if self.attrtype == 'relation':
            return self.value.agreeing_entities(entities=entities)

        elif self.attrtype == 'shape':
            return [entity for entity in entities if entity.shape.name == self.value]

        elif self.attrtype == 'color':
            return [entity for entity in entities if entity.color.name == self.value]

        elif self.attrtype == 'texture':
            return [entity for entity in entities if entity.texture.name == self.value]

        elif self.attrtype == 'combination':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) == self.value]

        elif self.attrtype == 'shapes':
            return [entity for entity in entities if entity.shape.name in self.value]

        elif self.attrtype == 'colors':
            return [entity for entity in entities if entity.color.name in self.value]

        elif self.attrtype == 'textures':
            return [entity for entity in entities if entity.texture.name in self.value]

        elif self.attrtype == 'combinations':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) in self.value]

        elif self.attrtype == 'x-max':
            # min distance to second
            return [entity for entity in entities if all(entity.center.x * self.value - other.center.x * self.value > Settings.min_distance for other in entities)]

        elif self.attrtype == 'y-max':
            # min distance to second
            return [entity for entity in entities if all(entity.center.y * self.value - other.center.y * self.value > Settings.min_distance for other in entities)]

        elif self.attrtype == 'size-max':
            # min difference to second
            return [entity for entity in entities if all(entity.shape.area * self.value - other.shape.area * self.value > Settings.min_area for other in entities)]

        elif self.attrtype == 'shade-max':
            # min difference to second
            return [entity for entity in entities if all(entity.color.shade * self.value - other.color.shade * self.value > Settings.min_shade for other in entities)]

    def disagreeing_entities(self, entities):
        if self.attrtype == 'relation':
            return self.value.disagreeing_entities(entities=entities)

        elif self.attrtype == 'shape':
            return [entity for entity in entities if entity.shape.name != self.value]

        elif self.attrtype == 'color':
            return [entity for entity in entities if entity.color.name != self.value]

        elif self.attrtype == 'texture':
            return [entity for entity in entities if entity.texture.name != self.value]

        elif self.attrtype == 'combination':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) != self.value]

        elif self.attrtype == 'shapes':
            return [entity for entity in entities if entity.shape.name not in self.value]

        elif self.attrtype == 'colors':
            return [entity for entity in entities if entity.color.name not in self.value]

        elif self.attrtype == 'textures':
            return [entity for entity in entities if entity.texture.name not in self.value]

        elif self.attrtype == 'combinations':
            return [entity for entity in entities if (entity.shape.name, entity.color.name, entity.texture.name) not in self.value]

        elif self.attrtype == 'x-max':
            return [entity for entity in entities if any(other.center.x * self.value - entity.center.x * self.value > Settings.min_distance for other in entities)]

        elif self.attrtype == 'y-max':
            return [entity for entity in entities if any(other.center.y * self.value - entity.center.y * self.value > Settings.min_distance for other in entities)]

        elif self.attrtype == 'size-max':
            return [entity for entity in entities if any(other.shape.area * self.value - entity.shape.area * self.value > Settings.min_area for other in entities)]

        elif self.attrtype == 'shade-max':
            return [entity for entity in entities if any(other.color.shade * self.value - entity.color.shade * self.value > Settings.min_shade for other in entities)]
