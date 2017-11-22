from shapeworld import util
from shapeworld.captions import Existential
from shapeworld.captioners import WorldCaptioner


class ExistentialCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect restrictor
    # 2: incorrect body

    def __init__(self, restrictor_captioner, body_captioner, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
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
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1]))

    def set_realizer(self, realizer):
        if not super(ExistentialCaptioner, self).set_realizer(realizer=realizer):
            return False

        assert realizer.existential is not None
        return True

    def rpn_length(self):
        return self.restrictor_captioner.rpn_length() + self.body_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(ExistentialCaptioner, self).rpn_symbols() | {Existential.__name__}

    def sample_values(self, mode, correct, predication):
        assert predication.empty()

        if not super(ExistentialCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        predication = predication.copy()

        if not self.restrictor_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1), predication=predication):  # 1: incorrect restrictor
            return False

        if not self.body_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 2), predication=predication):  # 2: incorrect body
            return False

        return True

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

        rstr_predication = predication.sub_predication()
        rstr_body_predication = predication.sub_predication()
        body_predication = predication.sub_predication()

        body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
        if body is None:
            return None
        self.body_captioner.apply_caption_to_predication(caption=body, predication=body_predication)

        restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
        if restrictor is None:
            return None
        self.restrictor_captioner.apply_caption_to_predication(caption=restrictor, predication=rstr_predication)

        if not self.pragmatical_tautology and rstr_predication.equals(other=body_predication):
            return None

        return Existential(restrictor=restrictor, body=body)

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: correct
            rstr_predication, body_predication = self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 1:  # 1: incorrect restrictor
            rstr_predication = predication.sub_predication()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=rstr_body_predication)
            body_predication = predication.sub_predication()
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        elif self.incorrect_mode == 2:  # 2: incorrect body
            rstr_predication = predication.sub_predication()
            self.restrictor_captioner.apply_caption_to_predication(caption=caption.restrictor, predication=rstr_predication)
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            if not self.body_captioner.incorrect(caption=caption.body, predication=rstr_body_predication, world=world):
                return False
            body_predication = predication.sub_predication()
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        if not self.pragmatical_tautology and rstr_predication.equals(other=body_predication):
            return False

        return True

    def apply_caption_to_predication(self, caption, predication):
        rstr_predication = predication.sub_predication()
        self.restrictor_captioner.apply_caption_to_predication(caption=caption.restrictor, predication=rstr_predication)
        rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=rstr_body_predication)
        body_predication = predication.sub_predication()
        self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)
        return rstr_predication, body_predication
