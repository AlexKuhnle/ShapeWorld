from itertools import combinations
from random import choice, random, shuffle
from shapeworld import util
from shapeworld.captions import Attribute, EntityType
from shapeworld.captioners import WorldCaptioner


class SingleAttributeTypeCaptioner(WorldCaptioner):

    def __init__(
        self,
        attribute,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        existing_attribute_rate=1.0
    ):
        super(SingleAttributeTypeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.attribute = attribute
        self.existing_attribute_rate = existing_attribute_rate

    def set_realizer(self, realizer):
        if not super(SingleAttributeTypeCaptioner, self).set_realizer(realizer):
            return False

        if self.attribute == 'shape':
            self.attributes = list(realizer.attributes.get('shape', ()))
        elif self.attribute == 'color':
            self.attributes = list(realizer.attributes.get('color', ()))
        elif self.attribute == 'texture':
            self.attributes = list(realizer.attributes.get('texture', ()))
        assert self.attributes

        return True

    def pn_length(self):
        return 2

    def pn_symbols(self):
        return super(SingleAttributeTypeCaptioner, self).pn_symbols() | \
            {EntityType.__name__ + '1'} | \
            {'{}-{}-{}'.format(Attribute.__name__, self.attribute, value) for value in self.attributes}

    def pn_arity(self):
        arity = super(SingleAttributeTypeCaptioner, self).pn_arity()
        arity[EntityType.__name__ + '1'] = 1
        arity.update({'{}-{}-{}'.format(Attribute.__name__, self.attribute, value): 0 for value in self.attributes})
        return arity

    def sample_values(self, mode, predication):
        if not super(SingleAttributeTypeCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        if len(self.attributes) <= 1:
            return False

        if (not self.logical_redundancy and predication.redundant(predicate=self.attribute)) or (not self.logical_contradiction and predication.blocked(predicate=self.attribute)):
            return False

        if not self.logical_tautology and predication.tautological(predicates=[self.attribute]):
            return False

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
        return util.merge_dicts(
            dict1=super(SingleAttributeTypeCaptioner, self).model(),
            dict2=dict(
                attributes=self.attributes,
                existing_attribute=self.existing_attribute
            )
        )

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        attributes = set()
        for entity in predication.agreeing:
            if self.attribute == 'shape' and entity.shape.name in self.attributes:
                attributes.add(entity.shape.name)
            elif self.attribute == 'color' and entity.color.name in self.attributes:
                attributes.add(entity.color.name)
            elif self.attribute == 'texture' and entity.texture.name in self.attributes:
                attributes.add(entity.texture.name)

        attribute = choice(list(attributes))
        if self.attribute == 'shape':
            attribute = Attribute(predtype='shape', value=attribute)
        elif self.attribute == 'color':
            attribute = Attribute(predtype='color', value=attribute)
        elif self.attribute == 'texture':
            attribute = Attribute(predtype='texture', value=attribute)

        if predication.contradictory(predicate=attribute):
            raise NotImplementedError
        elif not self.pragmatical_redundancy and predication.num_entities > 1 and predication.implies(predicate=attribute):
            raise NotImplementedError

        entity_type = EntityType(attributes=[attribute])

        if not self.correct(caption=entity_type, predication=predication):
            return None

        return entity_type

    def correct(self, caption, predication):
        for sub_predication in predication.get_sub_predications():
            if sub_predication.implies(predicate=caption) or sub_predication.implied_by(predicate=caption):
                return False

        return super().correct(caption=caption, predication=predication)

    def incorrect(self, caption, predication, world):
        if self.existing_attribute:
            assert len(caption.value) == 1 and caption.value[0].predtype == self.attribute
            if self.attribute == 'shape':
                attributes = list(set(entity.shape.name for entity in world.entities if entity.shape.name in self.attributes and entity.shape.name != caption.value[0].value))
            elif self.attribute == 'color':
                attributes = list(set(entity.color.name for entity in world.entities if entity.color.name in self.attributes and entity.color.name != caption.value[0].value))
            elif self.attribute == 'texture':
                attributes = list(set(entity.texture.name for entity in world.entities if entity.texture.name in self.attributes and entity.texture.name != caption.value[0].value))
        if not self.existing_attribute or len(attributes) == 0:
            if self.attribute == 'shape':
                attributes = self.shapes
            elif self.attribute == 'color':
                attributes = self.colors
            elif self.attribute == 'texture':
                attributes = self.textures
        caption.value[0] = Attribute(predtype=self.attribute, value=choice(attributes))

        return self.correct(caption=caption, predication=predication)
