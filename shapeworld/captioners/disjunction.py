from random import choice
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class DisjunctionCaptioner(WorldCaptioner):

    # correct modes
    # 0: incorrect (both incorrect)
    # 1: correct (first correct)
    # 2: correct (second correct)
    # 3: correct (both correct)

    def __init__(self, captioners, correct_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(DisjunctionCaptioner, self).__init__(
            internal_captioners=captioners,
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.correct_distribution = util.cumulative_distribution(util.value_or_default(correct_distribution, [1, 1, 1]))

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

    def sample_values(self, mode, correct, predication):
        assert predication.empty()

        if not super(DisjunctionCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.captioner1 = choice(self.internal_captioners)
        self.captioner2 = choice(self.internal_captioners)

        self.correct_mode = 0 if not correct else 1 + util.sample(self.correct_distribution)

        correct1 = (self.correct_mode == 1) or (self.correct_mode == 3)  # 1: first correct, 3: both correct
        correct2 = (self.correct_mode == 2) or (self.correct_mode == 3)  # 2: second correct, 3: both correct

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, correct=correct1, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, correct=correct2, predication=predication2):
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

        if self.correct_mode == 2:  # 2: second correct
            predication_copy = predication1.copy()
            clause1 = self.captioner1.caption(predication=predication_copy, world=world)
            if clause1 is None:
                return None
            if not self.captioner1.incorrect(caption=clause1, predication=predication1, world=world):
                return None

        else:
            clause1 = self.captioner1.caption(predication=predication1, world=world)
            if clause1 is None:
                return None

        predication2 = predication.sub_predication()

        if self.correct_mode == 1:  # 1: first correct
            predication_copy = predication2.copy()
            clause2 = self.captioner2.caption(predication=predication_copy, world=world)
            if clause2 is None:
                return None
            if not self.captioner2.incorrect(caption=clause2, predication=predication2, world=world):
                return None

        else:
            clause2 = self.captioner2.caption(predication=predication2, world=world)
            if clause2 is None:
                return None

        return Proposition(proptype='disjunction', clauses=(clause1, clause2))

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.correct_mode == 0:  # 0: both incorrect
            predication1 = predication.sub_predication()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            predication2 = predication.sub_predication()
            if self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False

        elif self.correct_mode == 1:  # 1: first correct
            predication1 = predication.sub_predication()
            self.captioner1.apply_caption_to_predication(caption=caption.clauses[0], predication=predication1)
            predication2 = predication.sub_predication()
            if self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False

        elif self.correct_mode == 2:  # 2: second correct
            predication1 = predication.sub_predication()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            predication2 = predication.sub_predication()
            self.captioner2.apply_caption_to_predication(caption=caption.clauses[1], predication=predication2)

        elif self.correct_mode == 3:  # 3: both correct
            self.apply_caption_to_predication(caption=caption, predication=predication)

        return True

    def apply_caption_to_predication(self, caption, predication):
        predication1 = predication.sub_predication()
        self.captioner1.apply_caption_to_predication(caption=caption.clauses[0], predication=predication1)
        predication2 = predication.sub_predication()
        self.captioner2.apply_caption_to_predication(caption=caption.clauses[1], predication=predication2)
