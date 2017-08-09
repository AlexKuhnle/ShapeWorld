from itertools import combinations
from random import choice
from shapeworld import util
from shapeworld.caption import Modifier, Noun, Relation
from shapeworld.captioners import WorldCaptioner


class AttributesRelationCaptioner(WorldCaptioner):

    # modes
    # 0: correct
    # 1: incorrect shape
    # 2: incorrect existing shape
    # 3: incorrect color
    # 4: incorrect existing color
    # 5: incorrect texture
    # 6: incorrect existing texture
    # 7: incorrect attributes
    # 8: incorrect existing attributes

    name = 'attributes_relation'
    statistics_header = 'hypernym,mode'

    def __init__(self, shapes=None, colors=None, textures=None, type_distribution=None, incorrect_distribution=None):
        super(AttributesRelationCaptioner, self).__init__()
        self.shapes = shapes
        self.colors = colors
        self.textures = textures
        self.type_distribution = util.cumulative_distribution(type_distribution or [1, 1, 1])
        self.incorrect_distribution = incorrect_distribution

    def set_realizer(self, realizer):
        if not super(AttributesRelationCaptioner, self).set_realizer(realizer):
            return False
        assert 'modifier' in realizer.relations and 'noun' in realizer.relations
        if self.shapes is None:
            self.shapes = list(realizer.modifiers.get('shape', ()))
            self.modifiers = list(('shape', value) for value in realizer.modifiers.get('shape', ()))
        else:
            self.shapes = list(value for value in realizer.modifiers.get('shape', ()) if value in self.shapes)
            self.modifiers = list(('shape', value) for value in realizer.modifiers.get('shape', ()) if value in self.shapes)
        if self.colors is None:
            self.colors = list(realizer.modifiers.get('color', ()))
            self.modifiers.extend(('color', value) for value in realizer.modifiers.get('color', ()))
        else:
            self.colors = list(value for value in realizer.modifiers.get('color', ()) if value in self.colors)
            self.modifiers.extend(('color', value) for value in realizer.modifiers.get('color', ()) if value in self.colors)
        if self.textures is None:
            self.textures = list(realizer.modifiers.get('texture', ()))
            self.modifiers.extend(('texture', value) for value in realizer.modifiers.get('texture', ()))
        else:
            self.textures = list(value for value in realizer.modifiers.get('texture', ()) if value in self.textures)
            self.modifiers.extend(('texture', value) for value in realizer.modifiers.get('texture', ()) if value in self.textures)
        assert self.shapes or self.colors or self.textures
        if self.incorrect_distribution is None:
            max_length = max(len(self.shapes), len(self.colors), len(self.textures))
            self.incorrect_distribution = util.cumulative_distribution([len(self.shapes), len(self.shapes), len(self.colors), len(self.colors), len(self.textures), len(self.textures), max_length, max_length])
        else:
            self.incorrect_distribution = util.cumulative_distribution(self.incorrect_distribution)
        return True

    def caption_world(self, entities, correct):
        if correct and len(entities) == 0:
            return None

        type1 = util.sample(self.type_distribution)

        if correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.incorrect_distribution)

        existing_shapes = set(entity.shape.name for entity in entities if entity.shape.name in self.shapes)
        existing_colors = set(entity.color.name for entity in entities if entity.color.name in self.colors)
        existing_textures = set(entity.texture.name for entity in entities if entity.texture.name in self.textures)

        attributes = list()

        if mode == 1 and len(self.shapes) <= 1:
            return None
        elif mode == 2 and len(existing_shapes) <= 1:
            return None
        elif self.shapes:
            shape_modifier = Modifier(modtype='shape', value='')
            attributes.append(shape_modifier)

        if mode == 3 and len(self.colors) <= 1:
            return None
        elif mode == 4 and len(existing_colors) <= 1:
            return None
        elif self.colors:
            color_modifier = Modifier(modtype='color', value='')
            attributes.append(color_modifier)

        if mode == 5 and len(self.textures) <= 1:
            return None
        elif mode == 6 and len(existing_textures) <= 1:
            return None
        elif self.textures:
            texture_modifier = Modifier(modtype='texture', value='')
            attributes.append(texture_modifier)

        for _ in range(AttributesRelationCaptioner.MAX_ATTEMPTS):

            entity = choice(entities)
            if self.shapes:
                shape_modifier.value = entity.shape.name
            if self.colors:
                color_modifier.value = entity.color.name
            if self.textures:
                texture_modifier.value = entity.texture.name

            if mode == 1:  # random shape
                shape_modifier.value = choice(list(value for value in self.shapes if value != shape_modifier.value))

            elif mode == 2:  # random existing shape
                shape_modifier.value = choice(list(value for value in existing_shapes if value != shape_modifier.value))

            elif mode == 3:  # random color
                color_modifier.value = choice(list(value for value in self.colors if value != color_modifier.value))

            elif mode == 4:  # random existing color
                color_modifier.value = choice(list(value for value in existing_colors if value != color_modifier.value))

            elif mode == 5:  # random texture
                texture_modifier.value = choice(list(value for value in self.textures if value != texture_modifier.value))

            elif mode == 6:  # random existing texture
                texture_modifier.value = choice(list(value for value in existing_textures if value != texture_modifier.value))

            elif mode == 7:  # random attributes
                if len(self.shapes) > 1:
                    shape_modifier.value = choice(list(value for value in self.shapes if value != shape_modifier.value))
                if len(self.colors) > 1:
                    color_modifier.value = choice(list(value for value in self.colors if value != color_modifier.value))
                if len(self.textures) > 1:
                    texture_modifier.value = choice(list(value for value in self.textures if value != texture_modifier.value))

            elif mode == 8:  # random existing attributes
                if len(existing_shapes) > 1:
                    shape_modifier.value = choice(list(value for value in existing_shapes if value != shape_modifier.value))
                if len(existing_colors) > 1:
                    color_modifier.value = choice(list(value for value in existing_colors if value != color_modifier.value))
                if len(existing_textures) > 1:
                    texture_modifier.value = choice(list(value for value in existing_textures if value != texture_modifier.value))

            if type1 == 0:  # modifier
                modifier = choice(attributes)
                if (modifier.modtype, modifier.value) not in self.modifiers:
                    continue
                relation = Relation(reltype='modifier', value=modifier)
            else:
                if type1 == 1:  # full noun
                    modifiers = attributes
                elif type1 == 2:  # hypernym
                    modifiers = choice([comb for n in range(len(attributes)) for comb in combinations(attributes, n)])
                noun = Noun(modifiers=modifiers)
                relation = Relation(reltype='noun', value=noun)

            if relation.agreement(entities=entities) == float(correct):
                self.report(type1, mode)
                return relation

        return None
