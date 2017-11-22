from itertools import combinations
from random import choice, random, shuffle
from shapeworld import util
from shapeworld.captions import Attribute, EntityType, Relation
from shapeworld.captioners import WorldCaptioner


class RegularAttributeCaptioner(WorldCaptioner):

    def __init__(self, existing_attribute_rate=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):

        super(RegularAttributeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.existing_attribute_rate = util.value_or_default(existing_attribute_rate, 0.5)

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

    def sample_values(self, mode, correct, predication):
        if not super(RegularAttributeCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        attributes = list()
        if len(self.shapes) > 1 and (self.logical_tautology or not predication.redundant(predicate='shape')):
            attributes.append('shape')
        if len(self.colors) > 1 and (self.logical_tautology or not predication.redundant(predicate='color')):
            attributes.append('color')
        if len(self.textures) > 1 and (self.logical_tautology or not predication.redundant(predicate='texture')):
            attributes.append('texture')

        if len(attributes) == 0:
            return False

        self.attribute = choice(attributes)

        self.existing_attribute = (correct or (random() < self.existing_attribute_rate and not predication.empty()))

        predication.apply(predicate=self.attribute)

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(RegularAttributeCaptioner, self).model(),
            dict2=dict(
                attribute=self.attribute,
                existing_attribute=self.existing_attribute
            )
        )

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        entity = predication.random_agreeing_entity()

        if self.attribute == 'shape':
            attribute = Attribute(predtype='shape', value=entity.shape.name)

        elif self.attribute == 'color':
            attribute = Attribute(predtype='color', value=entity.color.name)

        elif self.attribute == 'texture':
            attribute = Attribute(predtype='texture', value=entity.texture.name)

        if predication.contradictory(predicate=attribute):
            assert False
        elif not self.pragmatical_redundancy and predication.redundant(predicate=attribute):
            return None

        self.apply_caption_to_predication(caption=attribute, predication=predication)

        return attribute

    def incorrect(self, caption, predication, world):
        if not self.correct:

            if self.attribute == 'shape':  # random (existing) shape
                if self.existing_attribute:
                    values = util.unique_list(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes)
                else:
                    values = self.shapes

            elif self.attribute == 'color':  # random (existing) color
                if self.existing_attribute:
                    values = util.unique_list(entity.color.name for entity in world.entities if entity.color.name in self.colors)
                else:
                    values = self.colors

            elif self.attribute == 'texture':  # random (existing) texture
                if self.existing_attribute:
                    values = util.unique_list(entity.texture.name for entity in world.entities if entity.texture.name in self.textures)
                else:
                    values = self.textures

            caption.value = choice(values)

        self.apply_caption_to_predication(caption=caption, predication=predication)

        return True

    def apply_caption_to_predication(self, caption, predication):
        predication.apply(predicate=caption)
