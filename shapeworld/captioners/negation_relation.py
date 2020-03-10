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
        logical_redundancy_rate=0.0,
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

        assert 'negation' in realizer.relations and 1 in realizer.relations['negation']

        return True

    def pn_length(self):
        return self.relation_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(NegationRelationCaptioner, self).pn_symbols() | \
            {'{}-{}-{}'.format(Relation.__name__, 'negation', 1)}

    def pn_arity(self):
        arity = super(NegationRelationCaptioner, self).pn_arity()
        arity['{}-{}-{}'.format(Relation.__name__, 'negation', 1)] = 1
        return arity

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
        rel_predication = predication.copy(reset=True)
        if self.negation:
            relation = self.relation_captioner.caption(predication=rel_predication, world=world)
            if relation is None:
                return None
            rel_predication = predication.copy(reset=True)
            if not self.relation_captioner.incorrect(caption=relation, predication=rel_predication, world=world):
                return None

            relation = Relation(predtype='negation', value=1, reference=relation)

        else:
            relation = self.relation_captioner.caption(predication=rel_predication, world=world)
            if relation is None:
                return None

        if not self.correct(caption=relation, predication=predication):
            return None

        return relation

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: incorrect relation
            ref_predication = predication.copy(reset=True)
            if self.negation:
                caption.reference = self.relation_captioner.caption(predication=ref_predication, world=world)
                if caption.reference is None:
                    return False
            else:
                if not self.relation_captioner.incorrect(caption=caption, predication=ref_predication, world=world):
                    return False

        elif self.incorrect_mode == 1:  # 0: inverse negation
            if self.negation:
                caption.predtype = caption.reference.predtype
                caption.value = caption.reference.value
                caption.comparison = caption.reference.comparison
                caption.reference = caption.reference.reference
            else:
                caption.reference = Relation(predtype=caption.predtype, value=caption.value, reference=caption.reference, comparison=caption.comparison)
                caption.predtype = 'negation'
                caption.value = 1
                caption.comparison = None

        return self.correct(caption=caption, predication=predication)
