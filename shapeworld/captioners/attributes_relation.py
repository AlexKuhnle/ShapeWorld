from itertools import combinations
from random import choice, random
from shapeworld import util
from shapeworld.caption import Attribute, EntityType, Relation
from shapeworld.captioners import WorldCaptioner


class AttributesRelationCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect shape
    # 2: incorrect color
    # 3: incorrect texture
    # 4: incorrect attributes

    # attribute modes
    # 0: full type
    # 1: hypernym
    # 2: attribute

    def __init__(self, shapes=None, colors=None, textures=None, attribute_distribution=None, existing_attribute_ratio=None, incorrect_distribution=None, trivial_acceptance_rate=None):
        super(AttributesRelationCaptioner, self).__init__(trivial_acceptance_rate=trivial_acceptance_rate)
        self.shapes = shapes
        self.colors = colors
        self.textures = textures
        self.attribute_distribution = util.cumulative_distribution(util.value_or_default(attribute_distribution, [1, 1, 1]))
        self.existing_attribute_ratio = util.value_or_default(existing_attribute_ratio, 0.5)
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1, 1]))
        assert self.existing_attribute_ratio < 1.0 or self.attribute_distribution[0] > 0.0

    def set_realizer(self, realizer):
        if not super(AttributesRelationCaptioner, self).set_realizer(realizer):
            return False
        assert 'attribute' in realizer.relations and 'type' in realizer.relations
        if self.shapes is None:
            self.shapes = list(realizer.attributes.get('shape', ()))
            self.attributes = list(('shape', value) for value in realizer.attributes.get('shape', ()))
        else:
            self.shapes = list(value for value in realizer.attributes.get('shape', ()) if value in self.shapes)
            self.attributes = list(('shape', value) for value in realizer.attributes.get('shape', ()) if value in self.shapes)
        if self.colors is None:
            self.colors = list(realizer.attributes.get('color', ()))
            self.attributes.extend(('color', value) for value in realizer.attributes.get('color', ()))
        else:
            self.colors = list(value for value in realizer.attributes.get('color', ()) if value in self.colors)
            self.attributes.extend(('color', value) for value in realizer.attributes.get('color', ()) if value in self.colors)
        if self.textures is None:
            self.textures = list(realizer.attributes.get('texture', ()))
            self.attributes.extend(('texture', value) for value in realizer.attributes.get('texture', ()))
        else:
            self.textures = list(value for value in realizer.attributes.get('texture', ()) if value in self.textures)
            self.attributes.extend(('texture', value) for value in realizer.attributes.get('texture', ()) if value in self.textures)
        assert self.shapes or self.colors or self.textures
        if self.incorrect_distribution is None:
            max_length = max(len(self.shapes), len(self.colors), len(self.textures))
            self.incorrect_distribution = util.cumulative_distribution([len(self.shapes), len(self.shapes), len(self.colors), len(self.colors), len(self.textures), len(self.textures), max_length, max_length])
        else:
            self.incorrect_distribution = util.cumulative_distribution(self.incorrect_distribution)
        return True

    def sample_values(self, mode, correct):
        super(AttributesRelationCaptioner, self).sample_values(mode=mode, correct=correct)

        self.attribute_mode = util.sample(self.attribute_distribution)
        self.existing_attribute = correct or (random() < self.existing_attribute_ratio)
        while not correct and self.attribute_mode > 0 and self.existing_attribute:
            self.attribute_mode = util.sample(self.attribute_distribution)
            self.existing_attribute = correct or (random() < self.existing_attribute_ratio)
        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)
        while (self.incorrect_mode == 1 and len(self.shapes) <= 1) or (self.incorrect_mode == 2 and len(self.colors) <= 1) or (self.incorrect_mode == 3 and len(self.textures) <= 1):
            self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

    def model(self):
        return util.merge_dicts(
            dict1=super(AttributesRelationCaptioner, self).model(),
            dict2=dict(attribute_mode=self.attribute_mode, existing_attribute=self.existing_attribute, incorrect_mode=self.incorrect_mode)
        )

    def caption_world(self, entities, relevant_entities):
        if self.correct and len(relevant_entities) == 0:
            return None

        # problem if len(entities) == 0 but incorrect

        if self.existing_attribute and self.incorrect_mode > 0:
            shapes = set(entity.shape.name for entity in entities if entity.shape.name in self.shapes)
            colors = set(entity.color.name for entity in entities if entity.color.name in self.colors)
            textures = set(entity.texture.name for entity in entities if entity.texture.name in self.textures)
        else:
            shapes = self.shapes
            colors = self.colors
            textures = self.textures

        entity = choice(relevant_entities)
        attributes = list()

        if len(shapes) <= 1 and self.incorrect_mode == 1:
            return None
        elif len(self.shapes) > 1:
            shape_attribute = Attribute(attrtype='shape', value=entity.shape.name)
            attributes.append(shape_attribute)

        if len(colors) <= 1 and self.incorrect_mode == 2:
            return None
        elif len(self.colors) > 1:
            color_attribute = Attribute(attrtype='color', value=entity.color.name)
            attributes.append(color_attribute)

        if len(textures) <= 1 and self.incorrect_mode == 3:
            return None
        elif len(self.textures) > 1:
            texture_attribute = Attribute(attrtype='texture', value=entity.texture.name)
            attributes.append(texture_attribute)

        if self.incorrect_mode == 4 and len(shapes) <= 1 and len(colors) <= 1 and len(textures) <= 1:
            return None

        if self.incorrect_mode == 1:  # random (existing) shape
            shape_attribute.value = choice(list(value for value in shapes if value != shape_attribute.value))

        elif self.incorrect_mode == 2:  # random (existing) color
            color_attribute.value = choice(list(value for value in colors if value != color_attribute.value))

        elif self.incorrect_mode == 3:  # random (existing) texture
            texture_attribute.value = choice(list(value for value in textures if value != texture_attribute.value))

        elif self.incorrect_mode == 4:  # random (existing) attributes
            if len(shapes) > 1:
                shape_attribute.value = choice(list(shapes))
            if len(colors) > 1:
                color_attribute.value = choice(list(colors))
            if len(textures) > 1:
                texture_attribute.value = choice(list(textures))

        if self.attribute_mode == 2:  # attribute
            attribute = choice(attributes)
            if (attribute.attrtype, attribute.value) not in self.attributes:
                return None
            return Relation(reltype='attribute', value=attribute)
        else:
            if self.attribute_mode == 0:  # full type
                attributes = attributes
            elif self.attribute_mode == 1:  # hypernym
                attributes = choice([comb for n in range(len(attributes)) for comb in combinations(attributes, n)])
            etype = EntityType(attributes=attributes)
            return Relation(reltype='type', value=etype)
