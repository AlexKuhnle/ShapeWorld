from random import choice
from shapeworld import util
from shapeworld.captions import Quantifier
from shapeworld.captioners import WorldCaptioner


class QuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect restrictor
    # 2: incorrect body
    # 3: incorrect quantifier

    zero_quantifiers = {('count', 'lt', 0), ('count', 'leq', 0), ('count', 'eq', 0), ('count', 'eq-all', 0), ('count', 'lt', 1), ('ratio', 'lt', 0.0), ('ratio', 'leq', 0.0), ('ratio', 'eq', 0.0)}

    def __init__(self, restrictor_captioner, body_captioner, quantifiers=None, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(QuantifierCaptioner, self).__init__(
            internal_captioners=(restrictor_captioner, body_captioner),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.restrictor_captioner = restrictor_captioner
        self.body_captioner = body_captioner
        self.quantifiers = quantifiers
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 2]))

    def set_realizer(self, realizer):
        if not super(QuantifierCaptioner, self).set_realizer(realizer):
            return False

        if self.quantifiers is None:
            self.quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities]
        else:
            self.quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.quantifiers.items() if qtype in self.quantifiers for qrange, quantities in qranges.items() for quantity in quantities]

        return True

    def rpn_length(self):
        return self.restrictor_captioner.rpn_length() + self.body_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(QuantifierCaptioner, self).rpn_symbols() | {'{}-{}-{}-{}'.format(Quantifier.__name__, *quantifier) for quantifier in self.quantifiers}

    def sample_values(self, mode, correct, predication):
        assert predication.empty()

        if not super(QuantifierCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        self.qtype, self.qrange, self.quantity = choice(self.quantifiers)

        predication = predication.copy()

        if not self.restrictor_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1), predication=predication):  # 1: incorrect restrictor
            return False

        if (self.qtype, self.qrange, self.quantity) in QuantifierCaptioner.zero_quantifiers:
            # always incorrect body for zero quantification, since we need an incorrect body for a correct caption
            if not self.body_captioner.sample_values(mode=mode, correct=False, predication=predication):  # 2: incorrect body
                return False

        else:
            if not self.body_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 2), predication=predication):  # 2: incorrect body
                return False

        if self.incorrect_mode == 3:  # 3: incorrect quantifier
            self.incorrect_quantifiers = [(qtype, qrange, quantity) for qtype, qrange, quantity in self.quantifiers if qtype != self.qtype or qrange != self.qrange or quantity != self.quantity]

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(QuantifierCaptioner, self).model(),
            dict2=dict(
                qtype=self.qtype,
                qrange=self.qrange,
                quantity=self.quantity,
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

        if (self.qtype, self.qrange, self.quantity) in QuantifierCaptioner.zero_quantifiers:
            # special case: zero quantifier, hence incorrect body
            rstr_body_predication_copy = rstr_body_predication.copy()
            body = self.body_captioner.caption(predication=rstr_body_predication_copy, world=world)
            if body is None:
                return None
            if not self.body_captioner.incorrect(caption=body, predication=rstr_body_predication, world=world):
                return None

            restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication_copy, world=world)
            if restrictor is None:
                return None
            self.restrictor_captioner.apply_caption_to_predication(caption=restrictor, predication=rstr_body_predication)

        else:
            body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
            if body is None:
                return None

            restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
            if restrictor is None:
                return None

        self.restrictor_captioner.apply_caption_to_predication(caption=restrictor, predication=rstr_predication)
        self.body_captioner.apply_caption_to_predication(caption=body, predication=body_predication)

        if not self.pragmatical_tautology and rstr_predication.equals(other=body_predication):
            return None

        return Quantifier(qtype=self.qtype, qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, body=body)

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
            if (self.qtype, self.qrange, self.quantity) in QuantifierCaptioner.zero_quantifiers:
                # special case: zero quantifier, hence correct body
                caption.body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
                if caption.body is None:
                    return False
            else:
                if not self.body_captioner.incorrect(caption=caption.body, predication=rstr_body_predication, world=world):
                    return False
            body_predication = predication.sub_predication()
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        elif self.incorrect_mode == 3:  # 3: incorrect quantifier
            rstr_predication, body_predication = self.apply_caption_to_predication(caption=caption, predication=predication)
            caption.qtype, caption.qrange, caption.quantity = choice(self.quantifiers)

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
