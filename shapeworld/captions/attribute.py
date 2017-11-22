from shapeworld import util
from shapeworld.captions import Settings, Predicate


class Attribute(Predicate):

    predtypes = ('relation', 'shape', 'color', 'texture', 'combination', 'shapes', 'colors', 'textures', 'combinations', 'x-max', 'y-max', 'size-max', 'shade-max')

    def __init__(self, predtype, value):
        assert predtype in Attribute.predtypes
        if predtype == 'relation':
            from shapeworld.captions import Relation
            assert isinstance(value, Relation)
        elif predtype in ('shape', 'color', 'texture', 'combination'):
            assert isinstance(value, str)
        elif predtype in ('shapes', 'colors', 'textures', 'combinations'):
            assert isinstance(value, tuple) or isinstance(value, list)
            value = tuple(value)
        elif predtype in ('x-max', 'y-max', 'size-max', 'shade-max'):
            assert value == -1 or value == 1
        else:
            assert False
        super(Attribute, self).__init__(predtype=predtype, value=value)

    def model(self):
        if self.predtype == 'relation':
            value = self.value.model()
        elif self.predtype in ('shape', 'color', 'texture', 'combination', 'x-max', 'y-max', 'size-max', 'shade-max'):
            value = self.value
        elif self.predtype in ('shapes', 'colors', 'textures', 'combinations'):
            value = list(self.value)
        else:
            assert False
        return dict(
            component=str(self),
            predtype=self.predtype,
            value=value
        )

    def reverse_polish_notation(self):
        if self.predtype == 'relation':
            return self.value.reverse_polish_notation() + ['{}-{}'.format(self, self.predtype)]
        else:
            return ['{}-{}-{}'.format(self, self.predtype, self.value)]

    def pred_agreement(self, entity, predication):
        if self.predtype == 'relation':
            return self.value.pred_agreement(entity=entity, predication=predication)

        elif self.predtype == 'shape':
            return entity.shape.name == self.value

        elif self.predtype == 'color':
            return entity.color.name == self.value

        elif self.predtype == 'texture':
            return entity.texture.name == self.value

        elif self.predtype == 'combination':
            return (entity.shape.name, entity.color.name, entity.texture.name) == self.value

        elif self.predtype == 'shapes':
            return entity.shape.name in self.value

        elif self.predtype == 'colors':
            return entity.color.name in self.value

        elif self.predtype == 'textures':
            return entity.texture.name in self.value

        elif self.predtype == 'combinations':
            return (entity.shape.name, entity.color.name, entity.texture.name) in self.value

        elif self.predtype == 'x-max':
            return util.all_and_any((entity.center.x - other.center.x) * self.value > Settings.min_axis_distance for other in predication.agreeing if other != entity)

        elif self.predtype == 'y-max':
            return util.all_and_any((entity.center.y - other.center.y) * self.value > Settings.min_axis_distance for other in predication.agreeing if other != entity)

        elif self.predtype == 'size-max':
            return util.all_and_any((entity.shape.area - other.shape.area) * self.value > Settings.min_area for other in predication.agreeing if other != entity)

        elif self.predtype == 'shade-max':
            return util.all_and_any((entity.color.shade - other.color.shade) * self.value > Settings.min_shade for other in predication.agreeing if other != entity and other.color == entity.color)

    def pred_disagreement(self, entity, predication):
        if self.predtype == 'relation':
            return self.value.pred_disagreement(entity=entity, predication=predication)

        elif self.predtype == 'shape':
            return entity.shape.name != self.value

        elif self.predtype == 'color':
            return entity.color.name != self.value

        elif self.predtype == 'texture':
            return entity.texture.name != self.value

        elif self.predtype == 'combination':
            return (entity.shape.name, entity.color.name, entity.texture.name) != self.value

        elif self.predtype == 'shapes':
            return entity.shape.name not in self.value

        elif self.predtype == 'colors':
            return entity.color.name not in self.value

        elif self.predtype == 'textures':
            return entity.texture.name not in self.value

        elif self.predtype == 'combinations':
            return (entity.shape.name, entity.color.name, entity.texture.name) not in self.value

        elif self.predtype == 'x-max':
            return any((other.center.x - entity.center.x) * self.value > Settings.min_axis_distance for other in predication.not_disagreeing if other != entity)

        elif self.predtype == 'y-max':
            return any((other.center.y - entity.center.y) * self.value > Settings.min_axis_distance for other in predication.not_disagreeing if other != entity)

        elif self.predtype == 'size-max':
            return any((other.shape.area - entity.shape.area) * self.value > Settings.min_area for other in predication.not_disagreeing if other != entity)

        elif self.predtype == 'shade-max':
            return any((other.color.shade - entity.color.shade) * self.value > Settings.min_shade for other in predication.not_disagreeing if other != entity and other.color == entity.color)
