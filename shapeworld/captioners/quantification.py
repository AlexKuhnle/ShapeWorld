from random import choice, random, shuffle
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Modifier, Noun, Quantifier, Proposition


class QuantificationCaptioner(WorldCaptioner):

    MAX_ATTEMPTS = 10
    statistics_header = 'correct,mode,quantifier'

    def __init__(self, shapes, colors, textures, realizer=None, quantifier_tolerance=None, modes=None, quantifiers=None):
        # ideally requires modifiers of all values for modtype 'shape', 'color', 'texture'
        super().__init__(realizer=realizer, quantifier_tolerance=quantifier_tolerance)
        self.modes = cumulative_distribution(modes or [1, 1, 1, 1, 1])
        # self.incorrect_modes = cumulative_distribution(
        #     incorrect_mode_distribution or [1, 1, 1, 1, 1, 1])
        if quantifiers:
            self.quantifiers = self.realizer.get_quantifiers(names=quantifiers)
        else:
            self.quantifiers = self.realizer.get_quantifiers()
        self.shape_modifiers = [value for _, value in self.realizer.get_modifiers(modtypes=('shape',))]
        self.color_modifiers = [value for _, value in self.realizer.get_modifiers(modtypes=('color',))]
        # self.texture_modifiers = [value for _, value in self.realizer.get_modifiers(modtypes=('texture',))]

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
            if r < self.modes[0]:  # shape is [shape]
                mode = 1
                restrictor = Noun(predicates=())
                body = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)),))
            elif r < self.modes[1]:  # shape is [color]
                mode = 2
                restrictor = Noun(predicates=())
                body = Noun(predicates=(Modifier(modtype='color', value=choice(existing_colors)),))
            elif r < self.modes[2]:  # shape is [color] [shape]
                mode = 3
                restrictor = Noun(predicates=())
                body = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)), Modifier(modtype='color', value=choice(existing_colors))))
            elif r < self.modes[3]:   # [shape] is [color]
                mode = 4
                restrictor = Noun(predicates=(Modifier(modtype='shape', value=choice(existing_shapes)),))
                body = Noun(predicates=(Modifier(modtype='color', value=choice(existing_colors)),))
            elif r < self.modes[4]:  # [color] shape is [shape]
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
