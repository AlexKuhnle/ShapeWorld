from itertools import combinations
from random import choice, random, shuffle
from shapeworld import util
from shapeworld.captions import Attribute, EntityType
from shapeworld.captioners import WorldCaptioner


class RegularTypeCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect shape
    # 2: incorrect color
    # 3: incorrect texture
    # 4: incorrect attributes

    def __init__(self, hypernym_rate=None, existing_attribute_rate=None, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):

        super(RegularTypeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.hypernym_rate = util.value_or_default(hypernym_rate, 0.5)
        self.existing_attribute_rate = util.value_or_default(existing_attribute_rate, 0.5)
        self.incorrect_distribution = incorrect_distribution

    def set_realizer(self, realizer):
        if not super(RegularTypeCaptioner, self).set_realizer(realizer):
            return False

        self.shapes = list(realizer.attributes.get('shape', ()))
        self.colors = list(realizer.attributes.get('color', ()))
        self.textures = list(realizer.attributes.get('texture', ()))
        assert self.shapes or self.colors or self.textures

        if self.incorrect_distribution is None:
            # incorrect mode distribution uniform across attributes
            max_length = max(len(self.shapes), len(self.colors), len(self.textures)) - 1
            self.incorrect_distribution = util.cumulative_distribution([len(self.shapes) - 1, len(self.colors) - 1, len(self.textures) - 1, max_length])
        else:
            self.incorrect_distribution = util.cumulative_distribution(self.incorrect_distribution)

        return True

    def rpn_length(self):
        return 5

    def rpn_symbols(self):
        return super(RegularTypeCaptioner, self).rpn_symbols() | \
            set(str(n) for n in range(4)) | \
            {EntityType.__name__} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'shape', value) for value in self.shapes} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'color', value) for value in self.colors} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'texture', value) for value in self.textures}

    def sample_values(self, mode, correct, predication):
        if not super(RegularTypeCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        attributes = list()
        if len(self.shapes) > 1 and (self.logical_redundancy or not predication.redundant(predicate='shape')):
            attributes.append('shape')
        if len(self.colors) > 1 and (self.logical_redundancy or not predication.redundant(predicate='color')):
            attributes.append('color')
        if len(self.textures) > 1 and (self.logical_redundancy or not predication.redundant(predicate='texture')):
            attributes.append('texture')

        if not self.logical_tautology and predication.tautological(predicates=attributes):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)
            if (self.incorrect_mode != 1 or ('shape' in attributes and not predication.redundant(predicate='shape'))) and \
                    (self.incorrect_mode != 2 or ('color' in attributes and not predication.redundant(predicate='color'))) and \
                    (self.incorrect_mode != 3 or ('texture' in attributes and not predication.redundant(predicate='texture'))):
                break
        else:
            return False

        is_hypernym = 0
        if not self.logical_contradiction and self.incorrect_mode == 4:
            # since otherwise an incorrect predicate might contradict parts of the predication
            for attribute in list(attributes):
                if predication.redundant(predicate=attribute):
                    attributes.remove(attribute)
                    is_hypernym = 1  # attribute set is already smaller

        if len(attributes) == 0 and (not self.logical_tautology or not correct):
            return False

        self.hypernym = random() < self.hypernym_rate

        shuffle(attributes)

        if self.hypernym:
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
                self.attributes = choice([list(comb) for n in range(len(attributes) + is_hypernym) for comb in combinations(attributes, n)])
                if (not predication.tautological(predicates=self.attributes) or (self.logical_tautology and correct)) and \
                        (self.incorrect_mode != 1 or 'shape' in self.attributes) and \
                        (self.incorrect_mode != 2 or 'color' in self.attributes) and \
                        (self.incorrect_mode != 3 or 'texture' in self.attributes):
                    break
            else:
                return False

        else:
            self.attributes = attributes

        if self.incorrect_mode == 1:
            self.attributes.remove('shape')
            self.attributes.insert(0, 'shape')
        elif self.incorrect_mode == 2:
            self.attributes.remove('color')
            self.attributes.insert(0, 'color')
        elif self.incorrect_mode == 3:
            self.attributes.remove('texture')
            self.attributes.insert(0, 'texture')

        self.existing_attribute = (correct or (random() < self.existing_attribute_rate and (len(self.attributes) > 1 or not predication.empty())))

        for predtype in self.attributes:
            predication.apply(predicate=predtype)

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(RegularTypeCaptioner, self).model(),
            dict2=dict(
                hypernym=self.hypernym,
                attributes=self.attributes,
                existing_attribute=self.existing_attribute,
                incorrect_mode=self.incorrect_mode
            )
        )

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        entity = predication.random_agreeing_entity()
        attributes = list()

        for predtype in self.attributes:
            if predtype == 'shape':
                attributes.append(Attribute(predtype='shape', value=entity.shape.name))
            elif predtype == 'color':
                attributes.append(Attribute(predtype='color', value=entity.color.name))
            elif predtype == 'texture':
                attributes.append(Attribute(predtype='texture', value=entity.texture.name))

        for n in range(len(attributes) - 1, -1, -1):
            if predication.contradictory(predicate=attributes[n]):
                assert False
            elif not self.pragmatical_redundancy and predication.redundant(predicate=attributes[n]):
                attributes.pop(n)

        entity_type = EntityType(attributes=attributes)

        entity_type.apply_to_predication(predication=predication)

        return entity_type

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 1:  # random (existing) shape
            if self.existing_attribute:
                shapes = util.unique_list(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes)
            else:
                shapes = self.shapes
            for n, predicate in enumerate(caption.value):
                if predicate.predtype == 'shape':
                    caption.value[n] = Attribute(predtype='shape', value=choice(shapes))

        elif self.incorrect_mode == 2:  # random (existing) color
            if self.existing_attribute:
                colors = util.unique_list(entity.color.name for entity in world.entities if entity.color.name in self.colors)
            else:
                colors = self.colors
            for n, predicate in enumerate(caption.value):
                if predicate.predtype == 'color':
                    caption.value[n] = Attribute(predtype='color', value=choice(colors))

        elif self.incorrect_mode == 3:  # random (existing) texture
            if self.existing_attribute:
                textures = util.unique_list(entity.texture.name for entity in world.entities if entity.texture.name in self.textures)
            else:
                textures = self.textures
            for n, predicate in enumerate(caption.value):
                if predicate.predtype == 'texture':
                    caption.value[n] = Attribute(predtype='texture', value=choice(textures))

        elif self.incorrect_mode == 4:  # random (existing) attributes
            if self.existing_attribute:
                shapes = util.unique_list(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes)
                colors = util.unique_list(entity.color.name for entity in world.entities if entity.color.name in self.colors)
                textures = util.unique_list(entity.texture.name for entity in world.entities if entity.texture.name in self.textures)
            else:
                shapes = self.shapes
                colors = self.colors
                textures = self.textures
            for n, predicate in enumerate(caption.value):
                if predicate.predtype == 'shape' in self.attributes:
                    caption.value[n] = Attribute(predtype='shape', value=choice(shapes))
                elif predicate.predtype == 'color' in self.attributes:
                    caption.value[n] = Attribute(predtype='color', value=choice(colors))
                elif predicate.predtype == 'texture' in self.attributes:
                    caption.value[n] = Attribute(predtype='texture', value=choice(textures))

        caption.apply_to_predication(predication=predication)

        return True
