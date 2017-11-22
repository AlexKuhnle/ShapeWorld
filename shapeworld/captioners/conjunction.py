from random import choice
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class ConjunctionCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct (both correct)
    # 1: incorrect (first incorrect)
    # 2: incorrect (second incorrect)
    # 3: incorrect (both incorrect)

    def __init__(self, captioners, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(ConjunctionCaptioner, self).__init__(
            internal_captioners=captioners,
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(ConjunctionCaptioner, self).set_realizer(realizer=realizer):
            return False

        assert 'conjunction' in realizer.propositions
        return True

    def rpn_length(self):
        return super(ConjunctionCaptioner, self).rpn_length() * 2 + 2

    def rpn_symbols(self):
        return super(ConjunctionCaptioner, self).rpn_symbols() | \
            set(str(n) for n in range(1, 3)) | \
            {'{}-{}'.format(Proposition.__name__, 'conjunction')}

    def sample_values(self, mode, correct, predication):
        assert predication.empty()

        if not super(ConjunctionCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.captioner1 = choice(self.internal_captioners)
        self.captioner2 = choice(self.internal_captioners)

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        correct1 = (self.incorrect_mode != 1) and (self.incorrect_mode != 3)  # 1: first incorrect, 3: both incorrect
        correct2 = (self.incorrect_mode != 2) and (self.incorrect_mode != 3)  # 2: second incorrect, 3: both incorrect

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, correct=correct1, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, correct=correct2, predication=predication2):
            return False

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(ConjunctionCaptioner, self).model(),
            dict2=dict(
                incorrect_mode=self.incorrect_mode,
                captioner1=self.captioner1.model(),
                captioner2=self.captioner2.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        predication1 = predication.sub_predication()
        clause1 = self.captioner1.caption(predication=predication1, world=world)
        if clause1 is None:
            return None

        predication2 = predication.sub_predication()
        clause2 = self.captioner2.caption(predication=predication2, world=world)
        if clause2 is None:
            return None

        return Proposition(proptype='conjunction', clauses=(clause1, clause2))

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: both correct
            self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 1:  # 1: first incorrect
            predication1 = predication.sub_predication()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            predication2 = predication.sub_predication()
            self.captioner2.apply_caption_to_predication(caption=caption.clauses[1], predication=predication2)

        elif self.incorrect_mode == 2:  # 2: second incorrect
            predication1 = predication.sub_predication()
            self.captioner1.apply_caption_to_predication(caption=caption.clauses[0], predication=predication1)
            predication2 = predication.sub_predication()
            if self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False

        elif self.incorrect_mode == 3:  # 3: both incorrect
            predication1 = predication.sub_predication()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            predication2 = predication.sub_predication()
            if self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False

        return True

    def apply_caption_to_predication(self, caption, predication):
        predication1 = predication.sub_predication()
        self.captioner1.apply_caption_to_predication(caption=caption.clauses[0], predication=predication1)
        predication2 = predication.sub_predication()
        self.captioner2.apply_caption_to_predication(caption=caption.clauses[1], predication=predication2)
