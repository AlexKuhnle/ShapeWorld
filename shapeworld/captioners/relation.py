from random import choice
from shapeworld import util
from shapeworld.captions import Relation
from shapeworld.captioners import WorldCaptioner


class RelationCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect reference
    # 1: incorrect relation
    # 2: inverse relation

    def __init__(
        self,
        reference_captioner,
        comparison_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        relations=None,
        incorrect_distribution=(2, 1, 1)
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
            self.relations = [
                (predtype, value) for predtype, values in realizer.relations.items() for value in values
                if any((p == '*' or predtype == p) and (v == '*' or value == v) for p, v in self.relations)
            ]

        return True

    def pn_length(self):
        return self.reference_captioner.pn_length() + self.comparison_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(RelationCaptioner, self).pn_symbols() | {'{}-{}-{}'.format(Relation.__name__, *relation) for relation in self.relations}

    def pn_arity(self):
        arity = super(RelationCaptioner, self).pn_arity()
        arity.update({
            '{}-{}-{}'.format(Relation.__name__, *relation): 2 if relation[0] in Relation.ternary_relations else 1
            for relation in self.relations
        })
        return arity

    def sample_values(self, mode, predication):
        if not super(RelationCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.predtype, self.value = choice(self.relations)
            if self.predtype == 'size-rel' and not self.logical_contradiction and predication.blocked(predicate='shape'):
                continue
            elif self.predtype == 'shape-rel' and ((not self.logical_redundancy and predication.redundant(predicate='shape')) or (not self.logical_contradiction and predication.blocked(predicate='shape'))):
                continue
            elif self.predtype == 'shade-rel' and not self.logical_contradiction and predication.blocked(predicate='color'):
                continue
            elif self.predtype == 'color-rel' and ((not self.logical_redundancy and predication.redundant(predicate='color')) or (not self.logical_contradiction and predication.blocked(predicate='color'))):
                continue
            break
        else:
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            ref_predication = predication.copy(reset=True)
            if not self.reference_captioner.sample_values(mode=mode, predication=ref_predication):
                continue
            elif self.predtype == 'size-rel' and ((not predication.redundant(predicate='shape') and not ref_predication.redundant(predicate='shape')) or (not self.logical_redundancy and predication.redundant(predicate='shape') and ref_predication.redundant(predicate='shape')) or (not self.logical_contradiction and ref_predication.blocked(predicate='shape'))):
                continue
            elif self.predtype == 'shape-rel' and ((not self.logical_redundancy and ref_predication.redundant(predicate='shape')) or (not self.logical_contradiction and ref_predication.blocked(predicate='shape'))):
                continue
            elif self.predtype == 'shade-rel' and ((not predication.redundant(predicate='color') and not ref_predication.redundant(predicate='color')) or (not self.logical_redundancy and predication.redundant(predicate='color') and ref_predication.redundant(predicate='color')) or (not self.logical_contradiction and ref_predication.blocked(predicate='color'))):
                continue
            elif self.predtype == 'color-rel' and ((not self.logical_redundancy and ref_predication.redundant(predicate='color')) or (not self.logical_contradiction and ref_predication.blocked(predicate='color'))):
                continue
            break
        else:
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and not self.reference_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 2 and self.predtype in Relation.no_inverse_relations:
                continue
            elif self.incorrect_mode in (0, 2) and self.predtype == 'shape-rel' and not self.logical_contradiction and predication.redundant(predicate='shape'):
                continue
            elif self.incorrect_mode in (0, 2) and self.predtype == 'color-rel' and not self.logical_contradiction and predication.redundant(predicate='color'):
                continue
            break
        else:
            return False

        self.incorrect_predtype = self.predtype
        self.incorrect_value = self.value

        if self.incorrect_mode == 1:  # 1: incorrect relation
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
                self.incorrect_predtype, self.incorrect_value = choice(self.relations)
                if self.incorrect_predtype == self.predtype and self.incorrect_value == self.value:
                    continue
                elif self.incorrect_predtype == 'size-rel' and not self.logical_contradiction and (predication.blocked(predicate='shape') or ref_predication.blocked(predicate='shape')):
                    continue
                elif self.incorrect_predtype == 'shape-rel' and not self.logical_contradiction and (predication.redundant(predicate='shape') or ref_predication.redundant(predicate='shape')):
                    continue
                elif self.incorrect_predtype == 'shade-rel' and not self.logical_contradiction and (predication.blocked(predicate='color') or ref_predication.blocked(predicate='color')):
                    continue
                elif self.incorrect_predtype == 'color-rel' and not self.logical_contradiction and (predication.redundant(predicate='color') or ref_predication.redundant(predicate='color')):
                    continue
                break
            else:
                return False

        if self.predtype in Relation.ternary_relations or self.incorrect_predtype in Relation.ternary_relations:
            comp_predication = predication.copy(reset=True)
            if not self.comparison_captioner.sample_values(mode=mode, predication=comp_predication):
                return False

        if self.predtype in ('size-rel', 'shape-rel') or self.incorrect_predtype in ('size-rel', 'shape-rel'):
            predication.apply(predicate='shape')
            if not self.logical_redundancy or (not self.logical_contradiction and ref_predication.redundant(predicate='shape') and (self.incorrect_mode == 0 or self.incorrect_predtype in ('size-rel', 'shape-rel'))):
                predication.block(predicate='shape')
        elif self.predtype in ('shade-rel', 'color-rel') or self.incorrect_predtype in ('shade-rel', 'color-rel'):
            predication.apply(predicate='color')
            if not self.logical_redundancy or (not self.logical_contradiction and ref_predication.redundant(predicate='color') and (self.incorrect_mode == 0 or self.incorrect_predtype in ('shade-rel', 'color-rel'))):
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
        if self.incorrect_mode == 1:  # 1: incorrect relation
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
        ref_predication = predication.copy(reset=True)
        reference = self.reference_captioner.caption(predication=ref_predication, world=world)
        if reference is None:
            return None

        if self.predtype in Relation.ternary_relations or self.incorrect_predtype in Relation.ternary_relations:
            comp_predication = predication.copy(reset=True)
            comparison = self.comparison_captioner.caption(predication=comp_predication, world=world)
            if comparison is None:
                return None
            if comp_predication.implies(predicate=reference) or comp_predication.implied_by(predicate=reference):
                # reference and comparison should not overlap
                return None

        else:
            comparison = None

        relation = Relation(predtype=self.predtype, value=self.value, reference=reference, comparison=comparison)

        if not self.correct(caption=relation, predication=predication):
            return None

        return relation

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: incorrect reference
            ref_predication = predication.copy(reset=True)
            if not self.reference_captioner.incorrect(caption=caption.reference, predication=ref_predication, world=world):
                return False
            if self.predtype in Relation.ternary_relations:
                comp_predication = predication.copy(reset=True)
                if not self.comparison_captioner.correct(caption=caption.comparison, predication=comp_predication):
                    return False
                if comp_predication.implies(predicate=caption.reference) or comp_predication.implied_by(predicate=caption.reference):
                    # reference and comparison should not be equal
                    return False

        if self.incorrect_mode == 1:  # 1: incorrect relation
            caption.predtype = self.incorrect_predtype
            caption.value = self.incorrect_value

        elif self.incorrect_mode == 2:  # 2: inverse relation
            caption.value = -caption.value
            if (caption.predtype, caption.value) not in self.relations:
                return False

        return self.correct(caption=caption, predication=predication)
