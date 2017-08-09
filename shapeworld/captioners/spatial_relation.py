from random import choice
from shapeworld import WorldCaptioner, util
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Relation
from shapeworld.captioners import AttributesNounCaptioner


class SpatialRelationCaptioner(WorldCaptioner):

    # modes
    # 0: correct
    # 1: incorrect spatial relation
    # 2: inverse direction
    # 3: incorrect reference
    # 4: incorrect comparison

    name = 'spatial_relation'
    statistics_header = 'spatial,direction,mode'

    spatial_reltypes = ('x-rel', 'y-rel', 'z-rel', 'proximity-max', 'proximity-rel')

    def __init__(self, reference_captioner=None, relations=None, incorrect_distribution=None):
        assert relations is None or all(reltype in SpatialRelationCaptioner.spatial_reltypes for reltype in relations)
        super(SpatialRelationCaptioner, self).__init__()
        self.reference_captioner = reference_captioner or AttributesNounCaptioner()
        self.relations = relations
        self.incorrect_distribution = cumulative_distribution(incorrect_distribution or [1, 1, 1, 1])

    def set_realizer(self, realizer):
        super(SpatialRelationCaptioner, self).set_realizer(realizer)
        self.reference_captioner.set_realizer(realizer=realizer)
        if self.relations is None:
            self.relations = list((reltype, value) for reltype, values in realizer.relations.items() if reltype in SpatialRelationCaptioner.spatial_reltypes for value in values)
        else:
            self.relations = list((reltype, value) for reltype, values in realizer.relations.items() if reltype in self.relations for value in values)
        assert self.relations

    def caption_world(self, entities, correct):
        spatial, direction = choice(self.relations)

        if correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.incorrect_distribution)

        if correct and len(entities) <= 1 + (spatial in Relation.ternary_relations):
            return None
        if mode == 4 and spatial not in Relation.ternary_relations:
            return None

        for _ in range(SpatialRelationCaptioner.MAX_ATTEMPTS):

            if mode == 3:  # incorrect reference
                reference = self.reference_captioner(entities=entities, correct=False)
            else:
                reference = self.reference_captioner(entities=entities, correct=True)
            if reference is None:
                continue

            if mode == 4:  # incorrect comparison
                comparison = self.reference_captioner(entities=entities, correct=False)
            else:
                comparison = self.reference_captioner(entities=entities, correct=True)
            if comparison is None:
                continue

            if mode == 2:  # inverse direction
                if spatial in Relation.ternary_relations:
                    relation = Relation(reltype=spatial, value=(-direction), reference=reference, comparison=comparison)
                else:
                    relation = Relation(reltype=spatial, value=(-direction), reference=reference)

            else:
                if spatial in Relation.ternary_relations:
                    relation = Relation(reltype=spatial, value=direction, reference=reference, comparison=comparison)
                else:
                    relation = Relation(reltype=spatial, value=direction, reference=reference)

            if relation.agreement(entities=entities) == float(correct):
                self.report(spatial, direction, mode)
                return relation

        return None
