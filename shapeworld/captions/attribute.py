from shapeworld.captions import Predicate


class Attribute(Predicate):

    predtypes = ('relation', 'shape', 'color', 'texture')

    def __init__(self, predtype, value):
        assert predtype in Attribute.predtypes
        if predtype == 'relation':
            from shapeworld.captions import Relation
            assert isinstance(value, Relation)
        elif predtype in ('shape', 'color', 'texture'):
            assert isinstance(value, str)
        else:
            assert False
        super(Attribute, self).__init__(predtype=predtype, value=value)

    def model(self):
        if self.predtype == 'relation':
            value = self.value.model()
        elif self.predtype in ('shape', 'color', 'texture'):
            value = self.value
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

    def pred_disagreement(self, entity):
        if self.predtype == 'relation':
            return self.value.pred_disagreement(entity=entity)

        elif self.predtype == 'shape':
            return entity.shape.name != self.value

        elif self.predtype == 'color':
            return entity.color.name != self.value

        elif self.predtype == 'texture':
            return entity.texture.name != self.value
