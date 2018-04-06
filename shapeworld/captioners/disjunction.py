from copy import deepcopy
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class DisjunctionCaptioner(WorldCaptioner):

    # correct modes
    # 0: correct (first correct)
    # 1: correct (second correct)
    # 2: correct (both correct)

    def __init__(
        self,
        captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        correct_distribution=(1, 1, 1)
    ):
        super(DisjunctionCaptioner, self).__init__(
            internal_captioners=(captioner, deepcopy(captioner)),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.captioner1, self.captioner2 = self.internal_captioners
        self.correct_distribution = util.cumulative_distribution(correct_distribution)

    def set_realizer(self, realizer):
        if not super(DisjunctionCaptioner, self).set_realizer(realizer=realizer):
            return False

        assert 'disjunction' in realizer.propositions
        return True

    def rpn_length(self):
        return super(DisjunctionCaptioner, self).rpn_length() * 2 + 2

    def rpn_symbols(self):
        return super(DisjunctionCaptioner, self).rpn_symbols() | \
            set(str(n) for n in range(1, 3)) | \
            {'{}-{}'.format(Proposition.__name__, 'disjunction')}

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(DisjunctionCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        self.correct_mode = util.sample(self.correct_distribution)

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, predication=predication2):
            return False

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(DisjunctionCaptioner, self).model(),
            dict2=dict(
                correct_mode=self.correct_mode,
                captioner1=self.captioner1.model(),
                captioner2=self.captioner2.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        predication1 = predication.sub_predication()

        if self.correct_mode == 1:  # 1: second correct
            predication_copy = predication1.copy()
            clause1 = self.captioner1.caption(predication=predication_copy, world=world)
            if clause1 is None:
                return None
            if not self.captioner1.incorrect(caption=clause1, predication=predication1, world=world):
                return None
            predication_test = predication1.copy(include_sub_predications=True)
            if clause1.agreement(predication=predication_test, world=world) >= 0.0:
                return None

        else:
            clause1 = self.captioner1.caption(predication=predication1, world=world)
            if clause1 is None:
                return None

        predication2 = predication.sub_predication()

        if self.correct_mode == 0:  # 0: first correct
            predication_copy = predication2.copy()
            clause2 = self.captioner2.caption(predication=predication_copy, world=world)
            if clause2 is None:
                return None
            if not self.captioner2.incorrect(caption=clause2, predication=predication2, world=world):
                return None
            predication_test = predication2.copy(include_sub_predications=True)
            if clause2.agreement(predication=predication_test, world=world) >= 0.0:
                return None

        else:
            clause2 = self.captioner2.caption(predication=predication2, world=world)
            if clause2 is None:
                return None

        return Proposition(proptype='disjunction', clauses=(clause1, clause2))

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        predication1 = predication.sub_predication()
        if self.correct_mode == 1:
            caption.clauses[0].apply_to_predication(predication=predication1)
        elif not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
            return False

        predication2 = predication.sub_predication()
        if self.correct_mode == 0:
            caption.clauses[1].apply_to_predication(predication=predication2)
        elif not self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
            return False

        return True
