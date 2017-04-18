from copy import deepcopy
from itertools import product
from random import choice, random, shuffle
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Relation, Quantifier, Proposition


class SpatialCaptioner(WorldCaptioner):

    MAX_ATTEMPTS = 10
    statistics_header = 'correct,mode,relation,ref-hypernym,arg-hypernym'

    def __init__(self, shapes, colors, textures, realizer=None, quantifier_tolerance=None, incorrect_modes=None, hypernym_ratio=None):
        # ideally requires relations 'left', 'right', 'above', 'below'
        # requires quantifier ('absolute', 'geq', 1)
        # requires caption 'none'
        super(SpatialCaptioner, self).__init__(realizer=realizer, quantifier_tolerance=quantifier_tolerance)
        self.shapes = shapes
        self.colors = colors
        self.textures = textures
        self.incorrect_modes = cumulative_distribution(incorrect_modes or [2, 1, 1, 1, 1])
        self.hypernym_ratio = hypernym_ratio if hypernym_ratio is not None else 0.5
        self.spatial_relations = [reltype[0] for reltype in self.realizer.get_relations(reltypes=('left', 'right', 'above', 'below'))]

    def agreeing_arguments(self, entities):
        ref_entity = choice(entities)
        reference = self.realizer.noun_for_entity(entity=ref_entity)
        shuffle(self.spatial_relations)
        for reltype in self.spatial_relations:
            relation = Relation(reltype=reltype, reference=reference)
            if relation.agreeing_entities(entities=entities):
                break
        else:
            assert False
        arg_entity = choice(relation.agreeing_entities(entities))
        argument = self.realizer.noun_for_entity(entity=arg_entity)
        return ref_entity, reltype, arg_entity, reference, relation, argument

    def caption_world(self, world, correct):
        if correct and len(world['entities']) <= 1:
            return None
        entities = world['entities']
        existing_shapes = {entity['shape']['name'] for entity in entities}
        existing_colors = {entity['color']['name'] for entity in entities}
        # existing_textures = [entity['texture']['name'] for entity in entities]
        mode = 0

        for _ in range(SpatialCaptioner.MAX_ATTEMPTS):
            ref_entity, reltype, arg_entity, reference, relation, argument = self.agreeing_arguments(entities)

            if not correct:
                ref_entity = deepcopy(ref_entity)
                arg_entity = deepcopy(arg_entity)
                r = random()
                if r < self.incorrect_modes[0]:  # swap spatial relation
                    mode = 1
                    if reltype == 'left':
                        reltype = 'right'
                    elif reltype == 'right':
                        reltype = 'left'
                    elif reltype == 'above':
                        reltype = 'below'
                    elif reltype == 'below':
                        reltype = 'above'
                    if reltype not in self.spatial_relations:
                        continue
                elif r < self.incorrect_modes[1]:  # random reference attribute
                    mode = 2
                    r -= self.incorrect_modes[0]
                    r /= self.incorrect_modes[1] - self.incorrect_modes[0]
                    if r < 0.5:
                        ref_entity['shape']['name'] = choice([shape for shape in self.shapes if shape != ref_entity['shape']['name']])
                    else:
                        ref_entity['color']['name'] = choice([color for color in self.colors if color != ref_entity['color']['name']])
                elif r < self.incorrect_modes[2]:  # random existing reference attribute
                    mode = 3
                    r -= self.incorrect_modes[1]
                    r /= self.incorrect_modes[2] - self.incorrect_modes[1]
                    if r < 0.5:
                        if len(existing_shapes) == 1:
                            continue
                        ref_entity['shape']['name'] = choice([shape for shape in existing_shapes if shape != ref_entity['shape']['name']])
                    else:
                        if len(existing_colors) == 1:
                            continue
                        ref_entity['color']['name'] = choice([color for color in existing_colors if color != ref_entity['color']['name']])
                elif r < self.incorrect_modes[3]:  # random argument attribute
                    mode = 4
                    r -= self.incorrect_modes[2]
                    r /= self.incorrect_modes[3] - self.incorrect_modes[2]
                    if r < 0.5:
                        arg_entity['shape']['name'] = choice([shape for shape in self.shapes if shape != arg_entity['shape']['name']])
                    else:
                        arg_entity['color']['name'] = choice([color for color in self.colors if color != arg_entity['color']['name']])
                elif r < self.incorrect_modes[4]:  # random existing argument attribute
                    mode = 5
                    r -= self.incorrect_modes[3]
                    r /= self.incorrect_modes[4] - self.incorrect_modes[3]
                    if r < 0.5:
                        if len(existing_shapes) == 1:
                            continue
                        arg_entity['shape']['name'] = choice([shape for shape in existing_shapes if shape != arg_entity['shape']['name']])
                    else:
                        if len(existing_colors) == 1:
                            continue
                        arg_entity['color']['name'] = choice([color for color in existing_colors if color != arg_entity['color']['name']])
                reference = self.realizer.noun_for_entity(entity=ref_entity)
                relation = Relation(reltype=reltype, reference=reference)
                argument = self.realizer.noun_for_entity(entity=arg_entity)

            if self.hypernym_ratio:
                if random() < self.hypernym_ratio:
                    ref_hyp = True
                    ref_hypernyms = self.realizer.hypernyms_for_entity(entity=reference, noun=reference)
                else:
                    ref_hyp = False
                    ref_hypernyms = [reference]
                if random() < self.hypernym_ratio:
                    arg_hyp = True
                    arg_hypernyms = self.realizer.hypernyms_for_entity(entity=argument, noun=argument)
                else:
                    arg_hyp = False
                    arg_hypernyms = [argument]
                hypernyms = list(product(ref_hypernyms, arg_hypernyms))
                shuffle(hypernyms)
                for ref_hypernym, arg_hypernym in hypernyms:
                    rel_hypernym = Relation(reltype=reltype, reference=ref_hypernym)
                    quantifier = Quantifier(qtype='absolute', qrange='geq', quantity=1, tolerance=self.quantifier_tolerance, restrictor=arg_hypernym, body=rel_hypernym)
                    caption = Proposition(clauses=quantifier)
                    if caption.agreement(world=world) == float(correct):
                        self.report(correct, mode, reltype, ref_hyp, arg_hyp)
                        return caption

            quantifier = Quantifier(qtype='absolute', qrange='geq', quantity=1, tolerance=self.quantifier_tolerance, restrictor=argument, body=relation)
            caption = Proposition(clauses=quantifier)
            if caption.agreement(world=world) == float(correct):
                self.report(correct, mode, reltype, False, False)
                return caption

        return None
