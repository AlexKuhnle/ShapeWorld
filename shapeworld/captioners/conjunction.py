from copy import deepcopy
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class ConjunctionCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: first incorrect
    # 1: second incorrect
    # 2: both incorrect

    def __init__(
        self,
        captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        incorrect_distribution=(1, 1, 1)
    ):
        super(ConjunctionCaptioner, self).__init__(
            internal_captioners=(captioner, deepcopy(captioner)),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.captioner1, self.captioner2 = self.internal_captioners
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

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

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(ConjunctionCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, predication=predication2):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode in (0, 2) and not self.captioner1.incorrect_possible():
                continue
            elif self.incorrect_mode in (1, 2) and not self.captioner2.incorrect_possible():
                continue
            break
        else:
            return False

        return True

    def incorrect_possible(self):
        return self.captioner1.incorrect_possible() or self.captioner2.incorrect_possible()

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
        predication2 = predication.sub_predication()

        clause1 = self.captioner1.caption(predication=predication1, world=world)
        if clause1 is None:
            return None

        clause2 = self.captioner2.caption(predication=predication2, world=world)
        if clause2 is None:
            return None

        return Proposition(proptype='conjunction', clauses=(clause1, clause2))

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        predication1 = predication.sub_predication()
        predication2 = predication.sub_predication()

        if self.incorrect_mode == 0:  # 0: first incorrect
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            if caption.clauses[0].agreement(predication=predication1, world=world) >= 0.0:
                return False
            caption.clauses[1].apply_to_predication(predication=predication2)

        elif self.incorrect_mode == 1:  # 1: second incorrect
            caption.clauses[0].apply_to_predication(predication=predication1)
            if not self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False
            if caption.clauses[1].agreement(predication=predication2, world=world) >= 0.0:
                return False

        elif self.incorrect_mode == 2:  # 2: both incorrect
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            if caption.clauses[0].agreement(predication=predication1, world=world) >= 0.0:
                return False

            if not self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False
            if caption.clauses[1].agreement(predication=predication2, world=world) >= 0.0:
                return False

        return True
