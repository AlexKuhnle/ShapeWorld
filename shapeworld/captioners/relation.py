from random import choice
from shapeworld import util
from shapeworld.captions import Relation
from shapeworld.captioners import WorldCaptioner


class RelationCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect reference
    # 2: incorrect comparison
    # 3: incorrect relation
    # 4: inverse relation

    def __init__(self, reference_captioner, comparison_captioner, relations=None, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(RelationCaptioner, self).__init__(
            internal_captioners=(reference_captioner, comparison_captioner),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.reference_captioner = reference_captioner
        self.comparison_captioner = comparison_captioner
        self.relations = relations
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(RelationCaptioner, self).set_realizer(realizer):
            return False

        if self.relations is None:
            self.relations = [(predtype, value) for predtype, values in realizer.relations.items() for value in values]
        else:
            self.relations = [(predtype, value) for predtype, values in realizer.relations.items() if predtype in self.relations for value in values]

        return True

    def rpn_length(self):
        return self.reference_captioner.rpn_length() + self.comparison_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(RelationCaptioner, self).rpn_symbols() | {'{}-{}-{}'.format(Relation.__name__, *relation) for relation in self.relations}

    def sample_values(self, mode, correct, predication):
        if not super(RelationCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.predtype, self.value = choice(self.relations)

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)
            if (self.incorrect_mode != 2 or self.predtype in Relation.ternary_relations):
                # if incorrect comparison but relation not ternary
                break
        else:
            return False

        ref_predication = predication.copy(reset=True)
        if not self.reference_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1), predication=ref_predication):  # 1: incorrect reference
            return False

        comp_predication = predication.copy(reset=True)
        if not self.comparison_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 2), predication=comp_predication):  # 2: incorrect comparison
            return False

        if self.incorrect_mode == 3:  # 3: incorrect spatial relation
            self.incorrect_relations = [(predtype, value) for predtype, value in self.relations if predtype != self.predtype or value != self.value]

        predication.apply(predicate=self.predtype)

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(RelationCaptioner, self).model(),
            dict2=dict(
                predtype=self.predtype,
                value=self.value,
                incorrect_mode=self.incorrect_mode,
                reference_captioner=self.reference_captioner.model(),
                comparison_captioner=self.comparison_captioner.model()
            )
        )

    def caption(self, predication, world):
        ref_predication = predication.sub_predication(reset=True)
        reference = self.reference_captioner.caption(predication=ref_predication, world=world)
        if reference is None:
            return None

        if self.predtype in Relation.ternary_relations or self.incorrect_mode == 3:  # 3: incorrect relation
            comp_predication = predication.sub_predication(reset=True)
            comparison = self.comparison_captioner.caption(predication=comp_predication, world=world)
            if comparison is None:
                return None
            if ref_predication.equals(other=comp_predication):
                # reference and comparison should not be equal
                return None
        else:
            comparison = None

        relation = Relation(predtype=self.predtype, value=self.value, reference=reference, comparison=comparison)

        predication_copy = predication.copy(reset=True)
        predication.apply(predicate=relation, predication=predication_copy)

        return relation

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: correct
            ref_predication, comp_predication = self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 1:  # 1: incorrect reference
            ref_predication = predication.sub_predication(reset=True)
            if not self.reference_captioner.incorrect(caption=caption.reference, predication=ref_predication, world=world):
                return False
            if self.predtype in Relation.ternary_relations:
                comp_predication = predication.sub_predication(reset=True)
                self.comparison_captioner.apply_caption_to_predication(caption=caption.comparison, predication=comp_predication)
            predication_copy = predication.copy(reset=True)
            predication.apply(predicate=caption, predication=predication_copy)

        elif self.incorrect_mode == 2:  # 2: incorrect comparison
            ref_predication = predication.sub_predication(reset=True)
            self.reference_captioner.apply_caption_to_predication(caption=caption.reference, predication=ref_predication)
            comp_predication = predication.sub_predication(reset=True)
            if not self.comparison_captioner.incorrect(caption=caption.comparison, predication=comp_predication, world=world):
                return False
            predication_copy = predication.copy(reset=True)
            predication.apply(predicate=caption, predication=predication_copy)

        if self.incorrect_mode == 3:  # 3: incorrect relation
            caption.predtype, caption.value = choice(self.incorrect_relations)
            ref_predication, comp_predication = self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 4:  # 4: inverse relation
            caption.value = -caption.value
            if (caption.predtype, caption.value) not in self.relations:
                return False
            ref_predication, comp_predication = self.apply_caption_to_predication(caption=caption, predication=predication)

        if self.predtype in Relation.ternary_relations or self.incorrect_mode == 3:
            if ref_predication.equals(other=comp_predication):
                # reference and comparison should not be equal
                return False

        return True

    def apply_caption_to_predication(self, caption, predication):
        ref_predication = predication.sub_predication(reset=True)
        self.reference_captioner.apply_caption_to_predication(caption=caption.reference, predication=ref_predication)
        if caption.comparison is None:
            comp_predication = None
        else:
            comp_predication = predication.sub_predication(reset=True)
            self.comparison_captioner.apply_caption_to_predication(caption=caption.comparison, predication=comp_predication)
        predication_copy = predication.copy(reset=True)
        predication.apply(predicate=caption, predication=predication_copy)
        return ref_predication, comp_predication
