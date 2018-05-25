from random import choice, random
from shapeworld.captions import Attribute
from shapeworld.captioners import WorldCaptioner


class RegularAttributeCaptioner(WorldCaptioner):

    def __init__(
        self,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        existing_attribute_rate=0.5
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

    def rpn_length(self):
        return 1

    def rpn_symbols(self):
        return super(RegularAttributeCaptioner, self).rpn_symbols() | \
            {'{}-{}-{}'.format(Attribute.__name__, 'shape', value) for value in self.shapes} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'color', value) for value in self.colors} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'texture', value) for value in self.textures}

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

        entities = list()
        for entity in predication.agreeing:
            if self.attribute == 'shape':
                entities.append(entity.shape.name)
            elif self.attribute == 'color':
                entities.append(entity.color.name)
            elif self.attribute == 'texture':
                entities.append(entity.texture.name)

        entity = choice(entities)

        if self.attribute == 'shape':
            attribute = Attribute(predtype='shape', value=entity)

        elif self.attribute == 'color':
            attribute = Attribute(predtype='color', value=entity)

        elif self.attribute == 'texture':
            attribute = Attribute(predtype='texture', value=entity)

        if predication.contradictory(predicate=attribute):
            assert False
        elif not self.pragmatical_redundancy and predication.num_entities > 1 and predication.redundant(predicate=attribute):
            assert False
            return None

        attribute.apply_to_predication(predication=predication)

        return attribute

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

        caption.apply_to_predication(predication=predication)

        return True
