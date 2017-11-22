from random import choice
from shapeworld import util
from shapeworld.captions import ComparativeQuantifier
from shapeworld.captioners import WorldCaptioner


class ComparativeQuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect restrictor
    # 2: incorrect comparison
    # 3: incorrect body
    # 4: incorrect quantifier

    def __init__(self, restrictor_captioner, comparison_captioner, body_captioner, quantifiers=None, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(ComparativeQuantifierCaptioner, self).__init__(
            internal_captioners=(restrictor_captioner, comparison_captioner, body_captioner),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.restrictor_captioner = restrictor_captioner
        self.comparison_captioner = comparison_captioner
        self.body_captioner = body_captioner
        self.quantifiers = quantifiers
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1, 3]))

    def set_realizer(self, realizer):
        if not super(ComparativeQuantifierCaptioner, self).set_realizer(realizer):
            return False

        if self.quantifiers is None:
            self.quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.comparative_quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities]
        else:
            self.quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.comparative_quantifiers.items() if qtype in self.quantifiers for qrange, quantities in qranges.items() for quantity in quantities]

        return True

    def rpn_length(self):
        return self.restrictor_captioner.rpn_length() + self.comparison_captioner.rpn_length() + self.body_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(ComparativeQuantifierCaptioner, self).rpn_symbols() | {'{}-{}-{}-{}'.format(ComparativeQuantifier.__name__, *quantifier) for quantifier in self.quantifiers}

    def sample_values(self, mode, correct, predication):
        assert predication.empty()

        if not super(ComparativeQuantifierCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        self.qtype, self.qrange, self.quantity = choice(self.quantifiers)

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            rstr_predication = predication.copy()
            comp_predication = predication.copy()

            if not self.restrictor_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1), predication=rstr_predication):  # 1: incorrect restrictor
                return False

            if not self.comparison_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 2), predication=comp_predication):  # 2: incorrect comparison
                return False

            union_predication = rstr_predication.union(other=comp_predication)

            if self.body_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 3), predication=union_predication):  # 3: incorrect body
                break
        else:
            return False

        if self.incorrect_mode == 4:  # 4: incorrect quantifier
            self.incorrect_quantifiers = [(qtype, qrange, quantity) for qtype, qrange, quantity in self.quantifiers if qtype != self.qtype or qrange != self.qrange or quantity != self.quantity]

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(ComparativeQuantifierCaptioner, self).model(),
            dict2=dict(
                qtype=self.qtype,
                qrange=self.qrange,
                quantity=self.quantity,
                incorrect_mode=self.incorrect_mode,
                restrictor_captioner=self.restrictor_captioner.model(),
                comparison_captioner=self.comparison_captioner.model(),
                body_captioner=self.body_captioner.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        rstr_predication = predication.sub_predication()
        rstr_body_predication = predication.sub_predication()

        comp_predication = predication.sub_predication()
        comp_body_predication = predication.sub_predication()

        body_predication = predication.sub_predication()

        body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
        if body is None:
            return None
        self.body_captioner.apply_caption_to_predication(caption=body, predication=comp_body_predication)
        self.body_captioner.apply_caption_to_predication(caption=body, predication=body_predication)

        restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
        if restrictor is None:
            return None
        self.restrictor_captioner.apply_caption_to_predication(caption=restrictor, predication=rstr_predication)

        comparison = self.comparison_captioner.caption(predication=comp_body_predication, world=world)
        if comparison is None:
            return None
        self.comparison_captioner.apply_caption_to_predication(caption=comparison, predication=comp_predication)

        if rstr_predication.equals(other=comp_predication):
            # restrictor and comparison should not be equal
            return None

        if not self.pragmatical_tautology and (rstr_predication.equals(other=body_predication) or comp_predication.equals(other=body_predication)):
            return None

        return ComparativeQuantifier(qtype=self.qtype, qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, comparison=comparison, body=body)

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: correct
            rstr_predication, comp_predication, body_predication = self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 1:  # 1: incorrect restrictor
            rstr_predication = predication.sub_predication()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=rstr_body_predication)

            comp_predication = predication.sub_predication()
            self.comparison_captioner.apply_caption_to_predication(caption=caption.comparison, predication=comp_predication)
            comp_body_predication = predication.sub_predication(predication=comp_predication.copy())
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=comp_body_predication)

            body_predication = predication.sub_predication()
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        elif self.incorrect_mode == 2:  # 2: incorrect comparison
            rstr_predication = predication.sub_predication()
            self.restrictor_captioner.apply_caption_to_predication(caption=caption.restrictor, predication=rstr_predication)
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=rstr_body_predication)

            comp_predication = predication.sub_predication()
            if not self.comparison_captioner.incorrect(caption=caption.comparison, predication=comp_predication, world=world):
                return False
            comp_body_predication = predication.sub_predication(predication=comp_predication.copy())
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=comp_body_predication)

            body_predication = predication.sub_predication()
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        elif self.incorrect_mode == 3:  # 3: incorrect body
            rstr_predication = predication.sub_predication()
            self.restrictor_captioner.apply_caption_to_predication(caption=caption.restrictor, predication=rstr_predication)
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())

            comp_predication = predication.sub_predication()
            self.comparison_captioner.apply_caption_to_predication(caption=caption.comparison, predication=comp_predication)
            comp_body_predication = predication.sub_predication(predication=comp_predication.copy())

            body_predication = rstr_body_predication.union(other=comp_body_predication)
            if not self.body_captioner.incorrect(caption=caption.body, predication=body_predication, world=world):
                return False
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=rstr_body_predication)
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=comp_body_predication)

            body_predication = predication.sub_predication()
            self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        elif self.incorrect_mode == 4:  # 4: incorrect quantifier
            rstr_predication, comp_predication, body_predication = self.apply_caption_to_predication(caption=caption, predication=predication)
            caption.qtype, caption.qrange, caption.quantity = choice(self.quantifiers)

        if rstr_predication.equals(other=comp_predication):
            # restrictor and comparison should not be equal
            return None

        if not self.pragmatical_tautology and (rstr_predication.equals(other=body_predication) or comp_predication.equals(other=body_predication)):
            return None

        return True

    def apply_caption_to_predication(self, caption, predication):
        rstr_predication = predication.sub_predication()
        self.restrictor_captioner.apply_caption_to_predication(caption=caption.restrictor, predication=rstr_predication)
        rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
        self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=rstr_body_predication)

        comp_predication = predication.sub_predication()
        self.comparison_captioner.apply_caption_to_predication(caption=caption.comparison, predication=comp_predication)
        comp_body_predication = predication.sub_predication(predication=comp_predication.copy())
        self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=comp_body_predication)

        body_predication = predication.sub_predication()
        self.body_captioner.apply_caption_to_predication(caption=caption.body, predication=body_predication)

        return rstr_predication, comp_predication, body_predication
