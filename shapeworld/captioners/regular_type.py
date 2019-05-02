from itertools import combinations
from random import choice, random, shuffle
from shapeworld import util
from shapeworld.captions import Attribute, EntityType
from shapeworld.captioners import WorldCaptioner


class RegularTypeCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect shape
    # 1: incorrect color
    # 2: incorrect texture
    # 3: incorrect attributes

    def __init__(
        self,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        hypernym_rate=0.5,
        existing_attribute_rate=1.0,
        incorrect_distribution=(1, 1, 1, 1)
    ):
        super(RegularTypeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.hypernym_rate = hypernym_rate
        self.existing_attribute_rate = existing_attribute_rate
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(RegularTypeCaptioner, self).set_realizer(realizer):
            return False

        self.shapes = list(realizer.attributes.get('shape', ()))
        self.colors = list(realizer.attributes.get('color', ()))
        self.textures = list(realizer.attributes.get('texture', ()))
        assert self.shapes or self.colors or self.textures

        return True

    def pn_length(self):
        return 4

    def pn_symbols(self):
        return super(RegularTypeCaptioner, self).pn_symbols() | \
            {EntityType.__name__ + str(n) for n in range(0, 4)} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'shape', value) for value in self.shapes} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'color', value) for value in self.colors} | \
            {'{}-{}-{}'.format(Attribute.__name__, 'texture', value) for value in self.textures}

    def pn_arity(self):
        arity = super(RegularTypeCaptioner, self).pn_arity()
        arity.update({EntityType.__name__ + str(n): n for n in range(0, 4)})
        arity.update({'{}-{}-{}'.format(Attribute.__name__, 'shape', value): 0 for value in self.shapes})
        arity.update({'{}-{}-{}'.format(Attribute.__name__, 'color', value): 0 for value in self.colors})
        arity.update({'{}-{}-{}'.format(Attribute.__name__, 'texture', value): 0 for value in self.textures})
        return arity

    def sample_values(self, mode, predication):
        if not super(RegularTypeCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        self.valid_attributes = list()
        is_hypernym = 0
        if len(self.shapes) > 1:
            if (self.logical_redundancy or not predication.redundant(predicate='shape')) and (self.logical_contradiction or not predication.blocked(predicate='shape')):
                self.valid_attributes.append('shape')
            else:
                is_hypernym = 1
        if len(self.colors) > 1:
            if (self.logical_redundancy or not predication.redundant(predicate='color')) and (self.logical_contradiction or not predication.blocked(predicate='color')):
                self.valid_attributes.append('color')
            else:
                is_hypernym = 1
        if len(self.textures) > 1:
            if (self.logical_redundancy or not predication.redundant(predicate='texture')) and (self.logical_contradiction or not predication.blocked(predicate='texture')):
                self.valid_attributes.append('texture')
            else:
                is_hypernym = 1

        if not self.logical_tautology and predication.tautological(predicates=self.valid_attributes):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and ('shape' not in self.valid_attributes or (not self.logical_contradiction and predication.redundant(predicate='shape'))):
                continue
            elif self.incorrect_mode == 1 and ('color' not in self.valid_attributes or (not self.logical_contradiction and predication.redundant(predicate='color'))):
                continue
            elif self.incorrect_mode == 2 and ('texture' not in self.valid_attributes or (not self.logical_contradiction and predication.redundant(predicate='texture'))):
                continue
            elif self.incorrect_mode == 3 and (len(self.valid_attributes) == 0 or (not self.logical_contradiction and all(predication.redundant(predicate=attribute) for attribute in self.valid_attributes))):
                continue
            break
        else:
            return False

        if not self.logical_contradiction and self.incorrect_mode == 3:
            # since otherwise an incorrect predicate might contradict parts of the predication
            for attribute in list(self.valid_attributes):
                if predication.redundant(predicate=attribute):
                    self.valid_attributes.remove(attribute)
                    is_hypernym = 1  # attribute set is already smaller

        assert len(self.valid_attributes) > 0

        self.hypernym = random() < self.hypernym_rate

        shuffle(self.valid_attributes)

        if self.hypernym:
            hypernym_attributes = [list(comb) for n in range(1, len(self.valid_attributes) + int(is_hypernym)) for comb in combinations(self.valid_attributes, n)]
            shuffle(hypernym_attributes)
            for attributes in hypernym_attributes:
                if not self.logical_tautology and predication.tautological(predicates=attributes):
                    continue
                elif self.incorrect_mode == 0 and 'shape' not in attributes:
                    continue
                elif self.incorrect_mode == 1 and 'color' not in attributes:
                    continue
                elif self.incorrect_mode == 2 and 'texture' not in attributes:
                    continue
                self.attributes = attributes
                break
            else:
                return False

        else:
            self.attributes = list(self.valid_attributes)

        if self.incorrect_mode == 0:
            self.attributes.remove('shape')
            self.attributes.insert(0, 'shape')
        elif self.incorrect_mode == 1:
            self.attributes.remove('color')
            self.attributes.insert(0, 'color')
        elif self.incorrect_mode == 2:
            self.attributes.remove('texture')
            self.attributes.insert(0, 'texture')

        if self.existing_attribute_rate == 0.0:
            self.existing_attribute = False
        elif self.existing_attribute_rate == 1.0:
            self.existing_attribute = True
        else:
            self.existing_attribute = random() < self.existing_attribute_rate

        assert len(self.attributes) > 0

        for predtype in self.attributes:
            predication.apply(predicate=predtype)

        if self.incorrect_mode == 0:
            predication.block(predicate='shape')
        elif self.incorrect_mode == 1:
            predication.block(predicate='color')
        elif self.incorrect_mode == 2:
            predication.block(predicate='texture')
        elif self.incorrect_mode == 3:
            for predtype in self.attributes:
                predication.block(predicate=predtype)

        return True

    def incorrect_possible(self):
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

        entities = set()
        for entity in predication.agreeing:
            entity_attributes = list()
            for predtype in self.attributes:
                if predtype == 'shape' and entity.shape.name in self.shapes:
                    entity_attributes.append(entity.shape.name)
                elif predtype == 'color' and entity.color.name in self.colors:
                    entity_attributes.append(entity.color.name)
                elif predtype == 'texture' and entity.texture.name in self.textures:
                    entity_attributes.append(entity.texture.name)
                else:
                    break
            else:
                entity = tuple(entity_attributes)
                entities.add(entity)

        entity = choice(list(entities))

        attributes = list()
        for n, predtype in enumerate(self.attributes):
            if predtype == 'shape':
                attributes.append(Attribute(predtype='shape', value=entity[n]))
            elif predtype == 'color':
                attributes.append(Attribute(predtype='color', value=entity[n]))
            elif predtype == 'texture':
                attributes.append(Attribute(predtype='texture', value=entity[n]))

        for n in range(len(attributes) - 1, -1, -1):
            if predication.contradictory(predicate=attributes[n]):
                raise NotImplementedError
            elif not self.pragmatical_redundancy and predication.num_entities > 1 and predication.implies(predicate=attributes[n]):
                raise NotImplementedError
                attributes.pop(n)

        entity_type = EntityType(attributes=attributes)

        if not self.correct(caption=entity_type, predication=predication):
            return None

        return entity_type

    def correct(self, caption, predication):
        for sub_predication in predication.get_sub_predications():
            if sub_predication.implies(predicate=caption) or sub_predication.implied_by(predicate=caption):
                return False

        return super().correct(caption=caption, predication=predication)

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # random (existing) shape
            if self.existing_attribute:
                caption_shape = None
                for predicate in caption.value:
                    if predicate.predtype == 'shape':
                        caption_shape = predicate.value
                shapes = list(set(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes and entity.shape.name != caption_shape))
            if not self.existing_attribute or len(shapes) == 0:
                shapes = self.shapes
            if len(caption.value) == 0:
                caption.value.append(Attribute(predtype='shape', value=choice(shapes)))
            else:
                for n, predicate in enumerate(caption.value):
                    if predicate.predtype == 'shape':
                        caption.value[n] = Attribute(predtype='shape', value=choice(shapes))

        elif self.incorrect_mode == 1:  # random (existing) color
            if self.existing_attribute:
                caption_color = None
                for predicate in caption.value:
                    if predicate.predtype == 'color':
                        caption_color = predicate.value
                colors = list(set(entity.color.name for entity in world.entities if entity.color.name in self.colors and entity.color.name != caption_color))
            if not self.existing_attribute or len(colors) == 0:
                colors = self.colors
            if len(caption.value) == 0:
                caption.value.append(Attribute(predtype='color', value=choice(colors)))
            else:
                for n, predicate in enumerate(caption.value):
                    if predicate.predtype == 'color':
                        caption.value[n] = Attribute(predtype='color', value=choice(colors))

        elif self.incorrect_mode == 2:  # random (existing) texture
            if self.existing_attribute:
                caption_texture = None
                for predicate in caption.value:
                    if predicate.predtype == 'texture':
                        caption_texture = predicate.value
                textures = list(set(entity.texture.name for entity in world.entities if entity.texture.name in self.textures and entity.texture.name != caption_texture))
            if not self.existing_attribute or len(textures) == 0:
                textures = self.textures
            if len(caption.value) == 0:
                caption.value.append(Attribute(predtype='texture', value=choice(textures)))
            else:
                for n, predicate in enumerate(caption.value):
                    if predicate.predtype == 'texture':
                        caption.value[n] = Attribute(predtype='texture', value=choice(textures))

        elif self.incorrect_mode == 3:  # random (existing) attributes
            if self.existing_attribute:
                caption_shape = caption_color = caption_texture = None
                for predicate in caption.value:
                    if predicate.predtype == 'shape':
                        caption_shape = predicate.value
                    elif predicate.predtype == 'color':
                        caption_color = predicate.value
                    elif predicate.predtype == 'texture':
                        caption_texture = predicate.value
                shapes = list(set(entity.shape.name for entity in world.entities if entity.shape.name in self.shapes and entity.shape.name != caption_shape))
                colors = list(set(entity.color.name for entity in world.entities if entity.color.name in self.colors and entity.color.name != caption_color))
                textures = list(set(entity.texture.name for entity in world.entities if entity.texture.name in self.textures and entity.texture.name != caption_texture))
            if not self.existing_attribute or len(shapes) == 0:
                shapes = self.shapes
            if not self.existing_attribute or len(colors) == 0:
                colors = self.colors
            if not self.existing_attribute or len(textures) == 0:
                textures = self.textures
            if len(caption.value) == 0:
                attribute = choice(self.valid_attributes)
                if attribute == 'shape':
                    caption.value.append(Attribute(predtype='shape', value=choice(shapes)))
                elif attribute == 'color':
                    caption.value.append(Attribute(predtype='color', value=choice(colors)))
                elif attribute == 'texture':
                    caption.value.append(Attribute(predtype='texture', value=choice(textures)))
            else:
                for n, predicate in enumerate(caption.value):
                    if predicate.predtype == 'shape':
                        caption.value[n] = Attribute(predtype='shape', value=choice(shapes))
                    elif predicate.predtype == 'color':
                        caption.value[n] = Attribute(predtype='color', value=choice(colors))
                    elif predicate.predtype == 'texture':
                        caption.value[n] = Attribute(predtype='texture', value=choice(textures))

        return self.correct(caption=caption, predication=predication)
