from random import choice
from shapeworld import util
from shapeworld.caption import Relation
from shapeworld.captioners import WorldCaptioner, AttributesNounCaptioner


class ComparisonRelationCaptioner(WorldCaptioner):

    # modes
    # 0: correct
    # 1: incorrect comparison relation
    # 2: inverse comparison
    # 3: incorrect reference

    name = 'comparison_relation'
    statistics_header = 'comparison,relative,mode'

    comparison_reltypes = ('size-rel', 'shade-rel')

    def __init__(self, reference_captioner=None, relations=None, incorrect_distribution=None):
        assert relations is None or all(reltype in ComparisonRelationCaptioner.comparison_reltypes for reltype in relations)
        super(ComparisonRelationCaptioner, self).__init__()
        self.reference_captioner = reference_captioner or AttributesNounCaptioner()
        self.relations = relations
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution or [1, 1, 1])

    def set_realizer(self, realizer):
        if not super(ComparisonRelationCaptioner, self).set_realizer(realizer):
            return False
        self.reference_captioner.set_realizer(realizer=realizer)
        if self.relations is None:
            self.relations = list((reltype, value) for reltype, values in realizer.relations.items() if reltype in ComparisonRelationCaptioner.comparison_reltypes for value in values)
        else:
            self.relations = list((reltype, value) for reltype, values in realizer.relations.items() if reltype in self.relations for value in values)
        assert self.relations
        return True

    def caption_world(self, entities, correct):
        comparison, relative = choice(self.relations)

        if correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.incorrect_distribution)

        if correct and len(entities) <= 1:
            return None

        for _ in range(ComparisonRelationCaptioner.MAX_ATTEMPTS):

            if mode == 3:  # incorrect reference
                reference = self.reference_captioner(entities=entities, correct=False)
            else:
                reference = self.reference_captioner(entities=entities, correct=True)
            if reference is None:
                continue

            if mode == 2:  # inverse comparison
                if comparison in Relation.ternary_relations:
                    relation = Relation(reltype=comparison, value=(-relative), reference=reference, comparison=comparison)
                else:
                    relation = Relation(reltype=comparison, value=(-relative), reference=reference)

            else:
                relation = Relation(reltype=comparison, value=relative, reference=reference)

            if relation.agreement(entities=entities) == float(correct):
                self.report(comparison, relative, mode)
                return relation

        return None
