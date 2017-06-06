from random import choice, random, shuffle
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Modifier, Noun, Quantifier, Proposition


class QuantificationCaptioner(WorldCaptioner):

    name = 'quantification'
    statistics_header = 'correct,mode,quantifier'

    def __init__(self, shapes, colors, textures, quantifier_tolerance=None, mode_distribution=None, quantifiers=None):
        # ideally requires modifiers of all values for modtype 'shape', 'color', 'texture'
        super(QuantificationCaptioner, self).__init__(quantifier_tolerance=quantifier_tolerance)
        self.mode_distribution = cumulative_distribution(mode_distribution or [1, 1, 1, 1, 1])
        self.quantifiers = quantifiers
        # self.incorrect_distribution = cumulative_distribution(
        #     incorrect_mode_distribution or [1, 1, 1, 1, 1, 1])

    def set_realizer(self, realizer):
        if super(QuantificationCaptioner, self).set_realizer(realizer):
            if self.quantifiers:
                self.quantifiers = realizer.get_quantifiers(names=self.quantifiers)
            else:
                self.quantifiers = realizer.get_quantifiers()
            self.shape_modifiers = [value for _, value in realizer.get_modifiers(modtypes=('shape',))]
            self.color_modifiers = [value for _, value in realizer.get_modifiers(modtypes=('color',))]
            # self.texture_modifiers = [value for _, value in self.realizer.get_modifiers(modtypes=('texture',))]
            return True
        else:
            return False

    def caption_world(self, world, correct):
        existing_shapes = [entity['shape']['name'] for entity in world['entities']]
        existing_shapes = [shape for shape in existing_shapes if shape in self.shape_modifiers]
        existing_colors = [entity['color']['name'] for entity in world['entities']]
        existing_colors = [color for color in existing_colors if color in self.color_modifiers]
        # existing_textures = [entity['texture']['name'] for entity in entities]
        # existing_textures = [texture for texture in existing_textures if texture in self.texture_modifiers]
        mode = 0

        for _ in range(QuantificationCaptioner.MAX_ATTEMPTS):
            r = random()
            if r < self.mode_distribution[0]:  # shape is [shape]
                mode = 1
                restrictor = Noun(predicates=())
                body = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)),))
            elif r < self.mode_distribution[1]:  # shape is [color]
                mode = 2
                restrictor = Noun(predicates=())
                body = Noun(predicates=(Modifier(modtype='color', value=choice(existing_colors)),))
            elif r < self.mode_distribution[2]:  # shape is [color] [shape]
                mode = 3
                restrictor = Noun(predicates=())
                body = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)), Modifier(modtype='color', value=choice(existing_colors))))
            elif r < self.mode_distribution[3]:   # [shape] is [color]
                mode = 4
                restrictor = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)),))
                body = Noun(predicates=(Modifier(modtype='color', value=choice(existing_colors)),))
            elif r < self.mode_distribution[4]:  # [color] shape is [shape]
                mode = 5
                restrictor = Noun(predicates=(Modifier(modtype='color', value=choice(existing_colors)),))
                body = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)),))

            shuffle(self.quantifiers)
            for qtype, qrange, quantity in self.quantifiers:
                quantifier = Quantifier(qtype=qtype, qrange=qrange, quantity=quantity, tolerance=self.quantifier_tolerance, restrictor=restrictor, body=body)
                caption = Proposition(clauses=quantifier)
                if caption.agreement(world=world) == float(correct):
                    self.report(correct, mode, qtype, qrange, quantity)
                    return caption
