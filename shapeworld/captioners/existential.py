from shapeworld import util
from shapeworld.captions import Existential
from shapeworld.captioners import WorldCaptioner


class ExistentialCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect restrictor
    # 1: incorrect body

    def __init__(
        self,
        restrictor_captioner,
        body_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        incorrect_distribution=(1, 1)
    ):
        super(ExistentialCaptioner, self).__init__(
            internal_captioners=(restrictor_captioner, body_captioner),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.restrictor_captioner = restrictor_captioner
        self.body_captioner = body_captioner
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def pn_length(self):
        return self.restrictor_captioner.pn_length() + self.body_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(ExistentialCaptioner, self).pn_symbols() | {Existential.__name__}

    def pn_arity(self):
        arity = super(ExistentialCaptioner, self).pn_arity()
        arity[Existential.__name__] = 2
        return arity

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(ExistentialCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        # predication = predication.copy()

        if not self.body_captioner.sample_values(mode=mode, predication=predication):
            return False
        if not self.restrictor_captioner.sample_values(mode=mode, predication=predication):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and not self.restrictor_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 1 and not self.body_captioner.incorrect_possible():
                continue
            break
        else:
            return False

        return True

    def incorrect_possible(self):
        return self.restrictor_captioner.incorrect_possible() or self.body_captioner.incorrect_possible()

    def model(self):
        return util.merge_dicts(
            dict1=super(ExistentialCaptioner, self).model(),
            dict2=dict(
                incorrect_mode=self.incorrect_mode,
                restrictor_captioner=self.restrictor_captioner.model(),
                body_captioner=self.body_captioner.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        rstr_body_predication = predication.copy()

        body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
        if body is None:
            return None

        restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
        if restrictor is None:
            return None

        # also for incorrect
        # if not self.pragmatical_tautology and len(rstr_body_predication.agreeing) > 1 and (body_predication.equals(other=rstr_body_predication) or rstr_predication.equals(other=rstr_body_predication)):
        #     return None

        existential = Existential(restrictor=restrictor, body=body)

        if not self.correct(caption=existential, predication=predication):
            return None

        return existential

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: incorrect restrictor
            rstr_predication = predication.copy()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False

        elif self.incorrect_mode == 1:  # 1: incorrect body
            body_predication = predication.copy()
            if not self.body_captioner.incorrect(caption=caption.body, predication=body_predication, world=world):
                return False

        # if not self.pragmatical_tautology and rstr_predication.equals(other=body_predication):
        #     return False

        return self.correct(caption=caption, predication=predication)
