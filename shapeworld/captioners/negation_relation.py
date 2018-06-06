from random import random
from shapeworld import util
from shapeworld.captions import Relation
from shapeworld.captioners import WorldCaptioner


class NegationRelationCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect relation
    # 1: inverse negation

    def __init__(
        self,
        relation_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        incorrect_distribution=(1, 1)
    ):
        super(NegationRelationCaptioner, self).__init__(
            internal_captioners=(relation_captioner,),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.relation_captioner = relation_captioner
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(NegationRelationCaptioner, self).set_realizer(realizer):
            return False

        assert 'negation' in realizer.relations and -1 in realizer.relations['negation'] and 1 in realizer.relations['negation']

        return True

    def rpn_length(self):
        return self.relation_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(NegationRelationCaptioner, self).rpn_symbols() | \
            {'{}-{}-{}'.format(Relation.__name__, 'negation', -1), '{}-{}-{}'.format(Relation.__name__, 'negation', 1)}

    def sample_values(self, mode, predication):
        if not super(NegationRelationCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        self.incorrect_mode = util.sample(self.incorrect_distribution)

        self.negation = random() < 0.5

        return self.relation_captioner.sample_values(mode=mode, predication=predication)

    def incorrect_possible(self):
        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(NegationRelationCaptioner, self).model(),
            dict2=dict(
                negation=self.negation,
                incorrect_mode=self.incorrect_mode,
                relation_captioner=self.relation_captioner.model()
            )
        )

    def caption(self, predication, world):
        ref_predication = predication.sub_predication(reset=True)

        if self.negation:
            predication_copy = ref_predication.copy()
            reference = self.relation_captioner.caption(predication=predication_copy, world=world)
            if reference is None:
                return None
            if not self.relation_captioner.incorrect(caption=reference, predication=ref_predication, world=world):
                return None

        else:
            reference = self.relation_captioner.caption(predication=ref_predication, world=world)
            if reference is None:
                return None

        relation = Relation(predtype='negation', value=(-1 if self.negation else 1), reference=reference)

        predication.apply(predicate=relation, ref_predication=ref_predication)

        return relation

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: incorrect relation
            ref_predication = predication.sub_predication(reset=True)
            if self.negation:
                caption.reference = self.relation_captioner.caption(predication=ref_predication, world=world)
                if caption.reference is None:
                    return False
            else:
                if not self.relation_captioner.incorrect(caption=caption.reference, predication=ref_predication, world=world):
                    return False
            predication.apply(predicate=caption, ref_predication=ref_predication)

        elif self.incorrect_mode == 1:  # 0: inverse negation
            caption.value = -caption.value
            caption.apply_to_predication(predication=predication)

        return True
