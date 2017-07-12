from copy import deepcopy
from itertools import product
from random import choice, random, shuffle
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Relation, Quantifier, Proposition


class ComparisonCaptioner(WorldCaptioner):

    name = 'comparison'
    statistics_header = 'correct,mode,relation,value,ref-hypernym,arg-hypernym'

    def __init__(self, shapes, colors, textures, quantifier_tolerance=None, reltypes=None, incorrect_distribution=None, hypernym_ratio=None):
        # ideally requires relations 'size-rel', 'shade-rel'
        # requires quantifier ('absolute', 'geq', 1)
        # requires caption 'none'
        super(ComparisonCaptioner, self).__init__(quantifier_tolerance=quantifier_tolerance)
        self.shapes = shapes
        self.colors = colors
        self.textures = textures
        if reltypes is None:
            self.reltypes = ('size-rel', 'shade-rel')
        else:
            assert all(reltype in ('size-rel', 'shade-rel') for reltype in reltypes)
            self.reltypes = reltypes
        self.incorrect_distribution = cumulative_distribution(incorrect_distribution or [2, 2, 1, 1, 1, 1])
        self.hypernym_ratio = hypernym_ratio if hypernym_ratio is not None else 0.5

    def set_realizer(self, realizer):
        if super(ComparisonCaptioner, self).set_realizer(realizer):
            self.relations = realizer.get_relations(reltypes=self.reltypes)
            return True
        else:
            return False

    def caption_world(self, world, correct):
        if correct and len(world['entities']) <= 1:
            return None
        entities = world['entities']
        existing_shapes = {entity['shape']['name'] for entity in entities}
        existing_colors = {entity['color']['name'] for entity in entities}
        # existing_textures = [entity['texture']['name'] for entity in entities]
        mode = 0

        for _ in range(ComparisonCaptioner.MAX_ATTEMPTS):
            ref_entity = choice(entities)
            reference = self.realizer.noun_for_entity(entity=ref_entity)
            argument = None
            shuffle(self.relations)
            for reltype, value in self.relations:
                relation = Relation(reltype=reltype, value=value, reference=reference)
                if relation.agreeing_entities(entities=entities):
                    arg_entity = choice(relation.agreeing_entities(entities))
                    argument = self.realizer.noun_for_entity(entity=arg_entity)
                    break
            if argument is None:
                if not correct:
                    arg_entity = choice(entities)
                else:
                    continue

            if not correct:
                ref_entity = deepcopy(ref_entity)
                arg_entity = deepcopy(arg_entity)
                r = random()
                last_prob = 0.0
                for mode, prob in enumerate(self.incorrect_distribution, 1):
                    if r < prob:
                        r -= last_prob
                        r /= prob - last_prob
                        break
                    last_prob = prob

                if mode == 1:  # random comparison
                    reltype, value = choice(self.relations)
                if mode == 2:  # invert comparison
                    value = -value
                    if (reltype, value) not in self.relations:
                        continue
                elif mode == 3:  # random reference attribute
                    if r < 0.5:
                        ref_entity['shape']['name'] = choice([shape for shape in self.shapes if shape != ref_entity['shape']['name']])
                    else:
                        ref_entity['color']['name'] = choice([color for color in self.colors if color != ref_entity['color']['name']])
                elif mode == 4:  # random existing reference attribute
                    if r < 0.5:
                        if len(existing_shapes) == 1:
                            continue
                        ref_entity['shape']['name'] = choice([shape for shape in existing_shapes if shape != ref_entity['shape']['name']])
                    else:
                        if len(existing_colors) == 1:
                            continue
                        ref_entity['color']['name'] = choice([color for color in existing_colors if color != ref_entity['color']['name']])
                elif mode == 5:  # random argument attribute
                    if r < 0.5:
                        arg_entity['shape']['name'] = choice([shape for shape in self.shapes if shape != arg_entity['shape']['name']])
                    else:
                        arg_entity['color']['name'] = choice([color for color in self.colors if color != arg_entity['color']['name']])
                elif mode == 6:  # random existing argument attribute
                    if r < 0.5:
                        if len(existing_shapes) == 1:
                            continue
                        arg_entity['shape']['name'] = choice([shape for shape in existing_shapes if shape != arg_entity['shape']['name']])
                    else:
                        if len(existing_colors) == 1:
                            continue
                        arg_entity['color']['name'] = choice([color for color in existing_colors if color != arg_entity['color']['name']])
                reference = self.realizer.noun_for_entity(entity=ref_entity)
                relation = Relation(reltype=reltype, value=value, reference=reference)
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
                    rel_hypernym = Relation(reltype=reltype, value=value, reference=ref_hypernym)
                    quantifier = Quantifier(qtype='absolute', qrange='geq', quantity=1, tolerance=self.quantifier_tolerance, restrictor=arg_hypernym, body=rel_hypernym)
                    caption = Proposition(clauses=quantifier)
                    if caption.agreement(world=world) == float(correct):
                        self.report(correct, mode, reltype, value, ref_hyp, arg_hyp)
                        return caption

            quantifier = Quantifier(qtype='absolute', qrange='geq', quantity=1, tolerance=self.quantifier_tolerance, restrictor=argument, body=relation)
            caption = Proposition(clauses=quantifier)
            if caption.agreement(world=world) == float(correct):
                self.report(correct, mode, reltype, value, False, False)
                return caption

        return None
