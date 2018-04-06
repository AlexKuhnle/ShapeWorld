from copy import deepcopy
from shapeworld import util
from shapeworld.captions import Proposition
from shapeworld.captioners import WorldCaptioner


class ConjunctionCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect (first incorrect)
    # 1: incorrect (second incorrect)
    # 2: incorrect (both incorrect)

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

        self.incorrect_mode = util.sample(self.incorrect_distribution)

        predication1 = predication.copy()
        predication2 = predication.copy()

        if not self.captioner1.sample_values(mode=mode, predication=predication1):
            return False

        if not self.captioner2.sample_values(mode=mode, predication=predication2):
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

        if self.incorrect_mode == 0:  # 0: first incorrect
            predication1 = predication.sub_predication()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            predication2 = predication.sub_predication()
            caption.clauses[1].apply_to_predication(predication=predication2)

        elif self.incorrect_mode == 1:  # 1: second incorrect
            predication1 = predication.sub_predication()
            caption.clauses[0].apply_to_predication(predication=predication1)
            predication2 = predication.sub_predication()
            if not self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False

        elif self.incorrect_mode == 2:  # 2: both incorrect
            predication1 = predication.sub_predication()
            if not self.captioner1.incorrect(caption=caption.clauses[0], predication=predication1, world=world):
                return False
            predication_test = predication1.copy(include_sub_predications=True)
            if caption.clauses[0].agreement(predication=predication_test, world=world) >= 0.0:
                return False

            predication2 = predication.sub_predication()
            if not self.captioner2.incorrect(caption=caption.clauses[1], predication=predication2, world=world):
                return False
            predication_test = predication2.copy(include_sub_predications=True)
            if caption.clauses[1].agreement(predication=predication_test, world=world) >= 0.0:
                return False

        return True
