from random import choice
from shapeworld import util
from shapeworld.captions import Quantifier
from shapeworld.captioners import WorldCaptioner


class QuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect restrictor
    # 1: incorrect body
    # 2: incorrect quantifier

    def __init__(
        self,
        restrictor_captioner,
        body_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        quantifiers=None,
        incorrect_distribution=(1, 1, 2)
    ):
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
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(QuantifierCaptioner, self).set_realizer(realizer):
            return False

        if self.quantifiers is None:
            self.quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities]
        else:
            assert len(self.quantifiers) == 3
            self.quantifiers = [
                (qtype, qrange, quantity)
                for qtype, qranges in realizer.quantifiers.items() if self.quantifiers[0] is None or qtype in self.quantifiers[0]
                for qrange, quantities in qranges.items() if self.quantifiers[1] is None or qrange in self.quantifiers[1]
                for quantity in quantities if self.quantifiers[2] is None or (quantity >= 0 if self.quantifiers[2] == '+' else quantity in self.quantifiers[2])
            ]

        return True

    def rpn_length(self):
        return self.restrictor_captioner.rpn_length() + self.body_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(QuantifierCaptioner, self).rpn_symbols() | {'{}-{}-{}-{}'.format(Quantifier.__name__, *quantifier[:3 - int(quantifier[0] == 'composed')]) for quantifier in self.quantifiers}

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(QuantifierCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        self.incorrect_mode = util.sample(self.incorrect_distribution)

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.qtype, self.qrange, self.quantity = choice(self.quantifiers)
            if self.incorrect_mode == 0 and (self.qtype, self.qrange, self.quantity) in Quantifier.zero_quantifiers:
                continue
            elif self.incorrect_mode in (0, 1) and (self.qtype, self.qrange, self.quantity) in Quantifier.tautological_quantifiers:
                # always true in whatever way restrictor/body is changed
                continue
            break
        else:
            return False

        predication = predication.copy()

        if self.incorrect_mode == 0:  # 0: incorrect restrictor
            # incorrect after correct
            if (self.qtype, self.qrange, self.quantity) in Quantifier.zero_quantifiers:
                # always incorrect body for zero quantification, since we need an incorrect body for a correct caption
                if not self.body_captioner.sample_values(mode=mode, predication=predication):
                    return False
            else:
                if not self.body_captioner.sample_values(mode=mode, predication=predication):
                    return False
            if not self.restrictor_captioner.sample_values(mode=mode, predication=predication):
                return False

        else:
            if not self.restrictor_captioner.sample_values(mode=mode, predication=predication):
                return False
            if (self.qtype, self.qrange, self.quantity) in Quantifier.zero_quantifiers:
                # always incorrect body for zero quantification, since we need an incorrect body for a correct caption
                if not self.body_captioner.sample_values(mode=mode, predication=predication):
                    return False
            else:
                if not self.body_captioner.sample_values(mode=mode, predication=predication):
                    return False

        if self.incorrect_mode == 2:  # 2: incorrect quantifier
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
        body_predication = predication.sub_predication()
        rstr_body_predication = predication.sub_predication()

        if (self.qtype, self.qrange, self.quantity) in Quantifier.zero_quantifiers:
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
            restrictor.apply_to_predication(predication=rstr_body_predication)

        else:
            body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
            if body is None:
                return None

            restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
            if restrictor is None:
                return None

        restrictor.apply_to_predication(predication=rstr_predication)
        body.apply_to_predication(predication=body_predication)

        if self.quantity < 0 and -(self.quantity + 1) > rstr_predication.num_agreeing:
            return None

        if not self.pragmatical_tautology and (self.qtype, self.qrange, self.quantity) not in Quantifier.all_quantifiers and rstr_predication.equals(other=body_predication):
            # all quantification is inherently tautological
            return None

        return Quantifier(qtype=self.qtype, qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, body=body)

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: incorrect restrictor
            rstr_predication = predication.sub_predication()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False
            body_predication = predication.sub_predication()
            caption.body.apply_to_predication(predication=body_predication)
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            caption.body.apply_to_predication(predication=rstr_body_predication)

        elif self.incorrect_mode == 1:  # 1: incorrect body
            rstr_predication = predication.sub_predication()
            caption.restrictor.apply_to_predication(predication=rstr_predication)
            body_predication = predication.sub_predication()
            rstr_body_predication = predication.sub_predication(predication=rstr_predication.copy())
            if (self.qtype, self.qrange, self.quantity) in Quantifier.zero_quantifiers:
                # special case: zero quantifier, hence correct body
                caption.body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
                if caption.body is None:
                    return False
            else:
                if not self.body_captioner.incorrect(caption=caption.body, predication=rstr_body_predication, world=world):
                    return False
            caption.body.apply_to_predication(predication=body_predication)

        elif self.incorrect_mode == 2:  # 2: incorrect quantifier
            caption.qtype, caption.qrange, caption.quantity = choice(self.quantifiers)
            rstr_predication, body_predication, _ = caption.apply_to_predication(predication=predication)

        if caption.quantity < 0 and -(caption.quantity + 1) > rstr_predication.num_agreeing:
            return None

        if not self.pragmatical_tautology and (caption.qtype, caption.qrange, caption.quantity) not in Quantifier.all_quantifiers and rstr_predication.equals(other=body_predication):
            # all quantification is inherently tautological
            return False

        return True
