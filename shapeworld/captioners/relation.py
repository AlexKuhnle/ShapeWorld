from random import choice
from shapeworld import util
from shapeworld.captions import Relation
from shapeworld.captioners import WorldCaptioner


class RelationCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect reference
    # 1: incorrect comparison
    # 2: incorrect relation
    # 3: inverse relation

    def __init__(
        self,
        reference_captioner,
        comparison_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        relations=None,
        incorrect_distribution=(1, 1, 1, 1)
    ):
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
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(RelationCaptioner, self).set_realizer(realizer):
            return False

        if self.relations is None:
            self.relations = [(predtype, value) for predtype, values in realizer.relations.items() if predtype not in Relation.meta_relations for value in values]
        else:
            print(self.relations)
            self.relations = [
                (predtype, value) for predtype, values in realizer.relations.items() for value in values
                if any((p == '*' or predtype == p) and (v == '*' or value == v) for p, v in self.relations)
            ]

        return True

    def rpn_length(self):
        return self.reference_captioner.rpn_length() + self.comparison_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(RelationCaptioner, self).rpn_symbols() | {'{}-{}-{}'.format(Relation.__name__, *relation) for relation in self.relations}

    def sample_values(self, mode, predication):
        if not super(RelationCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        ref_predication = predication.copy(reset=True)
        if not self.reference_captioner.sample_values(mode=mode, predication=ref_predication):
            return False

        comp_predication = predication.copy(reset=True)
        if not self.comparison_captioner.sample_values(mode=mode, predication=comp_predication):
            return False

        self.predtype, self.value = choice(self.relations)

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and not self.reference_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 1 and not self.comparison_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 1 and self.predtype not in Relation.ternary_relations:
                # if incorrect comparison but relation not ternary
                continue
            break
        else:
            return False

        self.incorrect_predtype = self.predtype
        self.incorrect_value = self.value

        if self.incorrect_mode == 2:  # 2: incorrect relation
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
                self.incorrect_predtype, self.incorrect_value = choice(self.relations)
                if self.incorrect_predtype == self.predtype and self.incorrect_value == self.value:
                    continue
                break
            else:
                return False

        predication.apply(predicate=self.predtype)

        if (self.predtype == 'size-rel' or self.incorrect_predtype == 'size-rel') and ref_predication.redundant(predicate='shape'):
            predication.apply(predicate='shape')
        elif (self.predtype == 'shape-rel' or self.incorrect_predtype == 'shape-rel') and ref_predication.redundant(predicate='shape'):
            predication.block(predicate='shape')
        elif (self.predtype == 'shade-rel' or self.incorrect_predtype == 'shade-rel') and ref_predication.redundant(predicate='color'):
            predication.apply(predicate='color')
        elif (self.predtype == 'color-rel' or self.incorrect_predtype == 'color-rel') and ref_predication.redundant(predicate='color'):
            predication.block(predicate='color')

        return True

    def incorrect_possible(self):
        return True

    def model(self):
        model = super(RelationCaptioner, self).model()
        model.update(
            predtype=self.predtype,
            value=self.value,
            incorrect_mode=self.incorrect_mode,
            reference_captioner=self.reference_captioner.model()
        )
        if self.incorrect_mode == 2:  # 2: incorrect relation
            model.update(
                incorrect_predtype=self.incorrect_predtype,
                incorrect_value=self.incorrect_value
            )
        if self.predtype in Relation.ternary_relations or self.incorrect_predtype in Relation.ternary_relations:
            model.update(
                comparison_captioner=self.comparison_captioner.model()
            )
        return model

    def caption(self, predication, world):
        ref_predication = predication.sub_predication(reset=True)
        reference = self.reference_captioner.caption(predication=ref_predication, world=world)
        if reference is None:
            return None

        if self.predtype in Relation.ternary_relations or self.incorrect_predtype in Relation.ternary_relations:
            comp_predication = predication.sub_predication(reset=True)
            comparison = self.comparison_captioner.caption(predication=comp_predication, world=world)
            if comparison is None:
                return None
            if ref_predication.equals(other=comp_predication):
                # reference and comparison should not be equal
                return None
        else:
            comp_predication = None
            comparison = None

        relation = Relation(predtype=self.predtype, value=self.value, reference=reference, comparison=comparison)

        predication.apply(predicate=relation, ref_predication=ref_predication, comp_predication=comp_predication)

        return relation

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: incorrect reference
            ref_predication = predication.sub_predication(reset=True)
            if not self.reference_captioner.incorrect(caption=caption.reference, predication=ref_predication, world=world):
                return False
            if self.predtype in Relation.ternary_relations:
                comp_predication = predication.sub_predication(reset=True)
                caption.comparison.apply_to_predication(predication=comp_predication)
                if ref_predication.equals(other=comp_predication):
                    # reference and comparison should not be equal
                    return False
            else:
                comp_predication = None
            predication.apply(predicate=caption, ref_predication=ref_predication, comp_predication=comp_predication)

        elif self.incorrect_mode == 1:  # 1: incorrect comparison
            ref_predication = predication.sub_predication(reset=True)
            caption.reference.apply_to_predication(predication=ref_predication)
            comp_predication = predication.sub_predication(reset=True)
            if not self.comparison_captioner.incorrect(caption=caption.comparison, predication=comp_predication, world=world):
                return False
            if ref_predication.equals(other=comp_predication):
                # reference and comparison should not be equal
                return False
            predication.apply(predicate=caption, ref_predication=ref_predication, comp_predication=comp_predication)

        if self.incorrect_mode == 2:  # 2: incorrect relation
            caption.predtype = self.incorrect_predtype
            caption.value = self.incorrect_value
            ref_predication, comp_predication = caption.apply_to_predication(predication=predication)

        elif self.incorrect_mode == 3:  # 3: inverse relation
            caption.value = -caption.value
            if (caption.predtype, caption.value) not in self.relations:
                return False
            ref_predication, comp_predication = caption.apply_to_predication(predication=predication)

        return True
