from copy import deepcopy
from random import random
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class EquivalenceCaptioner(WorldCaptioner):

    def __init__(
        self,
        captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        both_correct_rate=0.5,
        first_only_correct_rate=0.5
    ):
        super(EquivalenceCaptioner, self).__init__(
            internal_captioners=(captioner, deepcopy(captioner)),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.captioner1, self.captioner2 = self.internal_captioners
        self.both_correct_rate = both_correct_rate
        self.first_only_correct_rate = first_only_correct_rate

    def set_realizer(self, realizer):
        if not super(EquivalenceCaptioner, self).set_realizer(realizer=realizer):
            return False

        assert 'equivalence' in realizer.propositions
        return True

    def pn_length(self):
        return super(EquivalenceCaptioner, self).pn_length() * 2 + 1

    def pn_symbols(self):
        return super(EquivalenceCaptioner, self).pn_symbols() | \
            {'{}-{}{}'.format(Proposition.__name__, 'equivalence', n) for n in range(2, 3)}

    def pn_arity(self):
        arity = super(EquivalenceCaptioner, self).pn_arity()
        arity.update({'{}-{}{}'.format(Proposition.__name__, 'equivalence', n): n for n in range(2, 3)})
        return arity

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(EquivalenceCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, predication=predication2):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.both_correct = random() < self.both_correct_rate
            self.first_only_correct = random() < self.first_only_correct_rate
            if not self.both_correct and (not self.captioner1.incorrect_possible() or not self.captioner2.incorrect_possible()):
                continue
            elif self.first_only_correct and not self.captioner2.incorrect_possible():
                continue
            elif not self.first_only_correct and not self.captioner1.incorrect_possible():
                continue
            break
        else:
            return False

        return True

    def incorrect_possible(self):
        return self.captioner1.incorrect_possible() or self.captioner2.incorrect_possible()

    def model(self):
        return util.merge_dicts(
            dict1=super(EquivalenceCaptioner, self).model(),
            dict2=dict(
                both_correct=self.both_correct,
                first_only_correct=self.first_only_correct,
                captioner1=self.captioner1.model(),
                captioner2=self.captioner2.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        predication1 = predication.copy()
        predication2 = predication1.sub_predication()

        if self.both_correct:
            clause2 = self.captioner2.caption(predication=predication2, world=world)
            if clause2 is None:
                return None

            clause1 = self.captioner1.caption(predication=predication1, world=world)
            if clause1 is None:
                return None

        else:
            clause2 = self.captioner2.caption(predication=predication2.copy(), world=world)
            if clause2 is None:
                return None
            if not self.captioner2.incorrect(caption=clause2, predication=predication2, world=world):
                return None
            if clause2.agreement(predication=predication2, world=world) >= 0.0:
                return None

            clause1 = self.captioner1.caption(predication=predication1.copy(), world=world)
            if clause1 is None:
                return None
            if not self.captioner1.incorrect(caption=clause1, predication=predication1, world=world):
                return None
            predication1.sub_predications.append(predication1.sub_predications.pop(0))
            if clause1.agreement(predication=predication1, world=world) >= 0.0:
                return None

        proposition = Proposition(proptype='equivalence', clauses=(clause1, clause2))

        if not self.correct(caption=proposition, predication=predication):
            return None

        return proposition

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.first_only_correct:
            if self.both_correct:
                predication2 = predication.copy()
                if not self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                    return False
                if caption.clauses[1].agreement(predication=predication2, world=world) >= 0.0:
                    return False

            else:
                predication1 = predication.copy()
                caption.clauses[0] = self.captioner1.caption(predication=predication1, world=world)
                if caption.clauses[0] is None:
                    return False

        else:
            if self.both_correct:
                predication1 = predication.copy()
                if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                    return False
                if caption.clauses[0].agreement(predication=predication1, world=world) >= 0.0:
                    return False

            else:
                predication2 = predication.copy()
                caption.clauses[1] = self.captioner2.caption(predication=predication2, world=world)
                if caption.clauses[1] is None:
                    return False

        return self.correct(caption=caption, predication=predication)
