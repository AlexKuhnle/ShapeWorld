from random import choice
from shapeworld import util
from shapeworld.captions import ComparativeQuantifier
from shapeworld.captioners import WorldCaptioner


class ComparativeQuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect restrictor
    # 1: incorrect comparison
    # 2: incorrect body
    # 3: incorrect quantifier

    def __init__(
        self,
        restrictor_captioner,
        comparison_captioner,
        body_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        comparative_quantifiers=None,
        incorrect_distribution=(1, 1, 1, 3)
    ):
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
        self.comparative_quantifiers = comparative_quantifiers
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(ComparativeQuantifierCaptioner, self).set_realizer(realizer):
            return False

        if self.comparative_quantifiers is None:
            self.comparative_quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.comparative_quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities]
        else:
            assert len(self.comparative_quantifiers) == 3
            self.comparative_quantifiers = [
                (qtype, qrange, quantity)
                for qtype, qranges in realizer.comparative_quantifiers.items() if self.comparative_quantifiers[0] is None or qtype in self.comparative_quantifiers[0]
                for qrange, quantities in qranges.items() if self.comparative_quantifiers[1] is None or qrange in self.comparative_quantifiers[1]
                for quantity in quantities if self.comparative_quantifiers[2] is None or quantity in self.comparative_quantifiers[2]
            ]

        return True

    def rpn_length(self):
        return self.restrictor_captioner.rpn_length() + self.comparison_captioner.rpn_length() + self.body_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(ComparativeQuantifierCaptioner, self).rpn_symbols() | {'{}-{}-{}-{}'.format(ComparativeQuantifier.__name__, *quantifier[:3 - int(quantifier[0] == 'composed')]) for quantifier in self.comparative_quantifiers}

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(ComparativeQuantifierCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        self.incorrect_mode = util.sample(self.incorrect_distribution)

        self.qtype, self.qrange, self.quantity = choice(self.comparative_quantifiers)

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            rstr_predication = predication.copy()
            comp_predication = predication.copy()

            if not self.restrictor_captioner.sample_values(mode=mode, predication=rstr_predication):
                return False

            if not self.comparison_captioner.sample_values(mode=mode, predication=comp_predication):
                return False

            union_predication = rstr_predication.union(other=comp_predication)

            if self.body_captioner.sample_values(mode=mode, predication=union_predication):
                break
        else:
            return False

        if self.incorrect_mode == 3:  # 3: incorrect quantifier
            self.incorrect_quantifiers = [(qtype, qrange, quantity) for qtype, qrange, quantity in self.comparative_quantifiers if qtype != self.qtype or qrange != self.qrange or quantity != self.quantity]

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
        body.apply_to_predication(predication=comp_body_predication)
        body.apply_to_predication(predication=body_predication)

        restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
        if restrictor is None:
            return None
        restrictor.apply_to_predication(predication=rstr_predication)

        comparison = self.comparison_captioner.caption(predication=comp_body_predication, world=world)
        if comparison is None:
            return None
        comparison.apply_to_predication(predication=comp_predication)

        if self.quantity < 0 and -self.quantity > rstr_body_predication.num_agreeing:
            return None

        if rstr_predication.equals(other=comp_predication):
            # restrictor and comparison should not be equal
            return None

        if not self.pragmatical_tautology and (rstr_predication.equals(other=body_predication) or comp_predication.equals(other=body_predication)):
            return None

        return ComparativeQuantifier(qtype=self.qtype, qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, comparison=comparison, body=body)

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: incorrect restrictor
            rstr_predication = predication.sub_predication()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            caption.body.apply_to_predication(predication=rstr_body_predication)

            comp_predication = predication.sub_predication()
            caption.comparison.apply_to_predication(predication=comp_predication)
            comp_body_predication = predication.sub_predication(predication=comp_predication.copy())
            caption.body.apply_to_predication(predication=comp_body_predication)

            body_predication = predication.sub_predication()
            caption.body.apply_to_predication(predication=body_predication)

        elif self.incorrect_mode == 1:  # 1: incorrect comparison
            rstr_predication = predication.sub_predication()
            caption.restrictor.apply_to_predication(predication=rstr_predication)
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            caption.body.apply_to_predication(predication=rstr_body_predication)

            comp_predication = predication.sub_predication()
            if not self.comparison_captioner.incorrect(caption=caption.comparison, predication=comp_predication, world=world):
                return False
            comp_body_predication = predication.sub_predication(predication=comp_predication.copy())
            caption.body.apply_to_predication(predication=comp_body_predication)

            body_predication = predication.sub_predication()
            caption.body.apply_to_predication(predication=body_predication)

        elif self.incorrect_mode == 2:  # 2: incorrect body
            rstr_predication = predication.sub_predication()
            caption.restrictor.apply_to_predication(predication=rstr_predication)
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())

            comp_predication = predication.sub_predication()
            caption.comparison.apply_to_predication(predication=comp_predication)
            comp_body_predication = predication.sub_predication(predication=comp_predication.copy())

            body_predication = rstr_body_predication.union(other=comp_body_predication)
            if not self.body_captioner.incorrect(caption=caption.body, predication=body_predication, world=world):
                return False
            caption.body.apply_to_predication(predication=rstr_body_predication)
            caption.body.apply_to_predication(predication=comp_body_predication)

            body_predication = predication.sub_predication()
            caption.body.apply_to_predication(predication=body_predication)

        elif self.incorrect_mode == 3:  # 3: incorrect quantifier
            caption.qtype, caption.qrange, caption.quantity = choice(self.comparative_quantifiers)
            rstr_predication, rstr_body_predication, comp_predication, _, body_predication = caption.apply_to_predication(predication=predication)

        if caption.quantity < 0 and -caption.quantity > rstr_body_predication.num_agreeing:
            return False

        if rstr_predication.equals(other=comp_predication):
            # restrictor and comparison should not be equal
            return False

        if not self.pragmatical_tautology and (rstr_predication.equals(other=body_predication) or comp_predication.equals(other=body_predication)):
            return False

        return True
