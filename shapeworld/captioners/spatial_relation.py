from random import choice
from shapeworld import util
from shapeworld.caption import Relation
from shapeworld.captioners import WorldCaptioner, AttributesTypeCaptioner


class SpatialRelationCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect spatial relation
    # 2: inverse direction
    # 3: incorrect reference
    # 4: incorrect comparison

    spatial_reltypes = ('x-rel', 'y-rel', 'z-rel', 'proximity-max', 'proximity-rel')

    def __init__(self, reference_captioner=None, comparison_captioner=None, relations=None, incorrect_distribution=None, trivial_acceptance_rate=None):
        assert relations is None or all(reltype in SpatialRelationCaptioner.spatial_reltypes for reltype in relations)
        self.reference_captioner = util.value_or_default(reference_captioner, AttributesTypeCaptioner())
        self.comparison_captioner = util.value_or_default(comparison_captioner, AttributesTypeCaptioner())
        super(SpatialRelationCaptioner, self).__init__(internal_captioners=(self.reference_captioner, self.comparison_captioner), trivial_acceptance_rate=trivial_acceptance_rate)
        self.relations = relations
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(SpatialRelationCaptioner, self).set_realizer(realizer):
            return False
        if self.relations is None:
            self.relations = list((reltype, value) for reltype, values in realizer.relations.items() if reltype in SpatialRelationCaptioner.spatial_reltypes for value in values)
        else:
            self.relations = list((reltype, value) for reltype, values in realizer.relations.items() if reltype in self.relations for value in values)
        assert self.relations
        return True

    def sample_values(self, mode, correct):
        super(SpatialRelationCaptioner, self).sample_values(mode=mode, correct=correct)

        self.reltype, self.value = choice(self.relations)
        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)
        while (self.incorrect_mode >= 2 and self.reltype == 'proximity-max') or (self.incorrect_mode == 4 and self.reltype not in Relation.ternary_relations):
            self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        self.reference_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 3))  # 3: incorrect reference
        self.comparison_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 4))  # 4: incorrect comparison

        if self.incorrect_mode == 1:  # 1: incorrect spatial relation
            self.incorrect_relations = [(reltype, value) for reltype, value in self.relations if reltype != self.reltype or value != self.value]

    def model(self):
        if self.reltype in Relation.ternary_relations:
            return util.merge_dicts(
                dict1=super(SpatialRelationCaptioner, self).model(),
                dict2=dict(reltype=self.reltype, value=self.value, incorrect_mode=self.incorrect_mode, reference_captioner=self.reference_captioner.model(), comparison_captioner=self.comparison_captioner.model())
            )
        else:
            return util.merge_dicts(
                dict1=super(SpatialRelationCaptioner, self).model(),
                dict2=dict(reltype=self.reltype, value=self.value, incorrect_mode=self.incorrect_mode, reference_captioner=self.reference_captioner.model())
            )

    def caption_world(self, entities, relevant_entities):
        if self.correct and len(entities) <= 1 + (self.reltype in Relation.ternary_relations):
            return None

        reference = self.reference_captioner(entities=entities)
        if reference is None:
            return None

        comparison = self.comparison_captioner(entities=entities)
        if comparison is None:
            return None

        relation = Relation(reltype=self.reltype, value=self.value, reference=reference, comparison=comparison)

        if self.incorrect_mode == 1:  # 1: incorrect spatial relation
            if relation.agreement(entities=relevant_entities) > 0.0:
                relation.reltype, relation.value = choice(self.incorrect_relations)
                return relation
            else:
                return None

        elif self.incorrect_mode == 2:  # 2: inverse direction
            if relation.agreement(entities=relevant_entities) > 0.0:
                relation.value = -relation.value
                return relation
            else:
                return None

        else:
            return relation
