from random import choice, random
from shapeworld.captions import Attribute
from shapeworld.captioners import WorldCaptioner


class RegularAttributeCaptioner(WorldCaptioner):

    def __init__(
        self,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        existing_attribute_rate=1.0
    ):
        super(RegularAttributeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.existing_attribute_rate = existing_attribute_rate

    def set_realizer(self, realizer):
        if not super(RegularAttributeCaptioner, self).set_realizer(realizer):
            return False

        self.shapes = list(realizer.attributes.get('shape', ()))
        self.colors = list(realizer.attributes.get('color', ()))
        self.textures = list(realizer.attributes.get('texture', ()))
        assert self.shapes or self.colors or self.textures

        return True

    def pn_length(self):
        return 1

    def pn_symbols(self):
        return super(RegularAttributeCaptioner, self).pn_symbols() | \
            {'{}-{}-{}'.format(Attribute.__name__, 'shape', value) for value in self.shapes} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'color', value) for value in self.colors} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'texture', value) for value in self.textures}

    def pn_arity(self):
        arity = super(RegularAttributeCaptioner, self).pn_arity()
        arity.update({'{}-{}-{}'.format(Attribute.__name__, 'shape', value): 0 for value in self.shapes})
        arity.update({'{}-{}-{}'.format(Attribute.__name__, 'color', value): 0 for value in self.colors})
        arity.update({'{}-{}-{}'.format(Attribute.__name__, 'texture', value): 0 for value in self.textures})
        return arity

    def sample_values(self, mode, predication):
        if not super(RegularAttributeCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        attributes = list()
        redundant_valid = self.logical_tautology and self.logical_contradition
        if len(self.shapes) > 1 and (redundant_valid or not predication.redundant(predicate='shape')):
            attributes.append('shape')
        if len(self.colors) > 1 and (redundant_valid or not predication.redundant(predicate='color')):
            attributes.append('color')
        if len(self.textures) > 1 and (redundant_valid or not predication.redundant(predicate='texture')):
            attributes.append('texture')

        if len(attributes) == 0:
            return False

        self.attribute = choice(attributes)

        if self.existing_attribute_rate == 0.0:
            self.existing_attribute = False
        elif self.existing_attribute_rate == 1.0:
            self.existing_attribute = True
        else:
            self.existing_attribute = random() < self.existing_attribute_rate

        predication.apply(predicate=self.attribute)
        predication.block(predicate=self.attribute)

        return True

    def incorrect_possible(self):
        return True

    def model(self):
        model = super(RegularAttributeCaptioner, self).model()
        model.update(
            attribute=self.attribute,
            existing_attribute=self.existing_attribute
        )
        return model

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        values = set()
        for entity in predication.agreeing:
            if self.attribute == 'shape' and entity.shape.name in self.shapes:
                values.add(entity.shape.name)
            elif self.attribute == 'color' and entity.color.name in self.colors:
                values.add(entity.color.name)
            elif self.attribute == 'texture' and entity.texture.name in self.textures:
                values.add(entity.texture.name)

        value = choice(list(values))

        if self.attribute == 'shape':
            attribute = Attribute(predtype='shape', value=value)

        elif self.attribute == 'color':
            attribute = Attribute(predtype='color', value=value)

        elif self.attribute == 'texture':
            attribute = Attribute(predtype='texture', value=value)

        if predication.contradictory(predicate=attribute):
            raise NotImplementedError
        elif not self.pragmatical_redundancy and predication.num_entities > 1 and predication.implies(predicate=attribute):
            raise NotImplementedError
            return None

        if not self.correct(caption=attribute, predication=predication):
            return None

        return attribute

    def correct(self, caption, predication):
        for sub_predication in predication.get_sub_predications():
            if sub_predication.implies(predicate=caption) or sub_predication.implied_by(predicate=caption):
                return False

        return super().correct(caption=caption, predication=predication)

    def incorrect(self, caption, predication, world):
        if self.attribute == 'shape':  # random (existing) shape
            if self.existing_attribute:
                values = list(set(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes and entity.shape.name != caption.value))
            if not self.existing_attribute or len(values) == 0:
                values = self.shapes

        elif self.attribute == 'color':  # random (existing) color
            if self.existing_attribute:
                values = list(set(entity.color.name for entity in world.entities if entity.color.name in self.colors and entity.color.name != caption.value))
            if not self.existing_attribute or len(values) == 0:
                values = self.colors

        elif self.attribute == 'texture':  # random (existing) texture
            if self.existing_attribute:
                values = list(set(entity.texture.name for entity in world.entities if entity.texture.name in self.textures and entity.texture.name != caption.value))
            if not self.existing_attribute or len(values) == 0:
                values = self.textures

        caption.value = choice(values)

        return self.correct(caption=caption, predication=predication)
