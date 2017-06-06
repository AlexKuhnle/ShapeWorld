from copy import deepcopy
from random import choice, random, shuffle
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Proposition


class ExistentialCaptioner(WorldCaptioner):

    name = 'existential'
    statistics_header = 'correct,mode,hypernym'

    def __init__(self, shapes, colors, textures, quantifier_tolerance=None, incorrect_distribution=None, hypernym_ratio=None):
        # requires relation 'existence'
        # requires quantifier ('absolute', 'geq', 1)
        # requires caption 'none'
        super(ExistentialCaptioner, self).__init__(quantifier_tolerance=quantifier_tolerance)
        self.shapes = shapes
        self.colors = colors
        self.textures = textures
        self.incorrect_distribution = cumulative_distribution(incorrect_distribution or [1, 1, 1, 1, 1, 1])
        self.hypernym_ratio = hypernym_ratio if hypernym_ratio is not None else 0.5

    def caption_world(self, world, correct):
        if correct and len(world['entities']) == 0:
            return None
        entities = world['entities']
        existing_shapes = {entity['shape']['name'] for entity in entities}
        existing_colors = {entity['color']['name'] for entity in entities}
        # existing_textures = [entity['texture']['name'] for entity in entities]
        mode = 0

        for _ in range(ExistentialCaptioner.MAX_ATTEMPTS):
            entity = choice(entities)

            if not correct:
                entity = deepcopy(entity)
                r = random()
                if r < self.incorrect_distribution[0]:  # random shape
                    mode = 1
                    entity['shape']['name'] = choice(self.shapes)
                elif r < self.incorrect_distribution[1]:  # random existing shape
                    if len(existing_shapes) == 1:
                        continue
                    mode = 2
                    entity['shape']['name'] = choice([shape for shape in existing_shapes if shape != entity['shape']['name']])
                elif r < self.incorrect_distribution[2]:  # random color
                    mode = 3
                    entity['color']['name'] = choice(self.colors)
                elif r < self.incorrect_distribution[3]:  # random existing color
                    if len(existing_colors) == 1:
                        continue
                    mode = 4
                    entity['color']['name'] = choice([color for color in existing_colors if color != entity['color']['name']])
                elif r < self.incorrect_distribution[4]:  # random shape and color
                    mode = 5
                    entity['shape']['name'] = choice(self.shapes)
                    entity['color']['name'] = choice(self.colors)
                elif r < self.incorrect_distribution[5]:  # random existing shape and color
                    mode = 6
                    entity['shape']['name'] = choice(list(existing_shapes))
                    entity['color']['name'] = choice(list(existing_colors))

            noun = self.realizer.noun_for_entity(entity=entity)
            # relation = Relation(reltype='existence')

            if self.hypernym_ratio and random() < self.hypernym_ratio:
                hypernyms = self.realizer.hypernyms_for_entity(entity=entity, noun=noun, include_universal=False)
                shuffle(hypernyms)
                for hypernym in hypernyms:
                    caption = Proposition(clauses=hypernym)
                    if caption.agreement(world=world) == float(correct):
                        self.report(correct, mode, True)
                        return caption

            caption = Proposition(clauses=noun)
            if caption.agreement(world=world) == float(correct):
                self.report(correct, mode, False)
                return caption

        return None
