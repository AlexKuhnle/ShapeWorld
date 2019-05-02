from shapeworld.captions import Predicate


class Attribute(Predicate):

    predtypes = {'relation', 'shape', 'color', 'texture', 'combination', 'shapes', 'colors', 'textures', 'combinations'}

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
        else:
            assert False
        super(Attribute, self).__init__(predtype=predtype, value=value)

    def model(self):
        if self.predtype == 'relation':
            value = self.value.model()
        elif self.predtype in ('shape', 'color', 'texture', 'combination'):
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

    def polish_notation(self, reverse=False):
        if self.predtype == 'relation':
            if reverse:
                return self.value.polish_notation(reverse=reverse) + ['{}-{}'.format(self, self.predtype)]
            else:
                return ['{}-{}'.format(self, self.predtype)] + self.value.polish_notation(reverse=reverse)
        else:
            return ['{}-{}-{}'.format(self, self.predtype, self.value)]

    def apply_to_predication(self, predication):
        predication.apply(predicate=self)

    def pred_agreement(self, entity):
        if self.predtype == 'relation':
            return self.value.pred_agreement(entity=entity)

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

    def pred_disagreement(self, entity):
        if self.predtype == 'relation':
            return self.value.pred_disagreement(entity=entity)

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
