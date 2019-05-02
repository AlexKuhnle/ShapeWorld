from copy import deepcopy
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class ImplicationCaptioner(WorldCaptioner):

    # correct modes (second => first)
    # 0: both correct
    # 1: first correct, second incorrect
    # 2: both incorrect

    def __init__(
        self,
        captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        correct_distribution=(1, 1, 1)
    ):
        super(ImplicationCaptioner, self).__init__(
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
        if not super(ImplicationCaptioner, self).set_realizer(realizer=realizer):
            return False

        assert 'implication' in realizer.propositions
        return True

    def pn_length(self):
        return super(ImplicationCaptioner, self).pn_length() * 2 + 1

    def pn_symbols(self):
        return super(ImplicationCaptioner, self).pn_symbols() | \
            {'{}-{}{}'.format(Proposition.__name__, 'implication', n) for n in range(2, 3)}

    def pn_arity(self):
        arity = super(ImplicationCaptioner, self).pn_arity()
        arity.update({'{}-{}{}'.format(Proposition.__name__, 'implication', n): n for n in range(2, 3)})
        return arity

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(ImplicationCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, predication=predication2):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.correct_mode = util.sample(self.correct_distribution)
            if self.correct_mode == 1 and not self.captioner2.incorrect_possible():
                continue
            if self.correct_mode == 2 and (not self.captioner1.incorrect_possible() or not self.captioner2.incorrect_possible()):
                continue
            break
        else:
            return False

        return True

    def incorrect_possible(self):
        return self.captioner1.incorrect_possible()

    def model(self):
        return util.merge_dicts(
            dict1=super(ImplicationCaptioner, self).model(),
            dict2=dict(
                correct_mode=self.correct_mode,
                captioner1=self.captioner1.model(),
                captioner2=self.captioner2.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        predication1 = predication.copy()
        predication2 = predication1.sub_predication()

        if self.correct_mode == 0:  # 0: both correct
            clause2 = self.captioner2.caption(predication=predication2, world=world)
            if clause2 is None:
                return None
        else:
            clause2 = self.captioner2.caption(predication=predication2.copy(), world=world)
            if clause2 is None:
                return None
            if not self.captioner2.incorrect(caption=clause2, predication=predication2, world=world):
                return None
            if clause2.agreement(predication=predication2, world=world) >= 0.0:
                return None

        if self.correct_mode == 2:  # 2: both incorrect
            clause1 = self.captioner1.caption(predication=predication1.copy(), world=world)
            if clause1 is None:
                return None
            if not self.captioner1.incorrect(caption=clause1, predication=predication1, world=world):
                return None
            predication1.sub_predications.append(predication1.sub_predications.pop(0))
            if clause1.agreement(predication=predication1, world=world) >= 0.0:
                return None
        else:
            clause1 = self.captioner1.caption(predication=predication1, world=world)
            if clause1 is None:
                return None

        proposition = Proposition(proptype='implication', clauses=(clause1, clause2))

        if not self.correct(caption=proposition, predication=predication):
            return None

        return proposition

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.correct_mode != 2:  # 2: both incorrect
            predication1 = predication.copy()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            if caption.clauses[0].agreement(predication=predication1, world=world) >= 0.0:
                return False

        if self.correct_mode != 0:  # 0: both correct
            predication2 = predication.copy()
            caption.clauses[1] = self.captioner2.caption(predication=predication2, world=world)
            if caption.clauses[1] is None:
                return False

        return self.correct(caption=caption, predication=predication)
