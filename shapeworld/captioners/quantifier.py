from random import choice, random
from shapeworld import util
from shapeworld.captions import Quantifier
from shapeworld.captioners import WorldCaptioner


class QuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect restrictor
    # 1: incorrect body
    # 2: closest quantity
    # 3: incorrect range
    # 4: incorrect quantity
    # 5: incorrect quantifier of same type

    def __init__(
        self,
        restrictor_captioner,
        body_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        quantifiers=None,
        incorrect_distribution=(3, 3, 3, 1, 1, 1),
        zero_quantification_rate=0.1,
        all_quantification_rate=0.3
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
        self.zero_quantification_rate = zero_quantification_rate
        self.all_quantification_rate = all_quantification_rate

    def set_realizer(self, realizer):
        if not super(QuantifierCaptioner, self).set_realizer(realizer):
            return False

        if self.quantifiers is None:
            self.quantifiers = [(qtype, qrange, quantity) for qtype, qranges in realizer.quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities]
        else:
            self.quantifiers = Quantifier.filter(quantifiers=((qtype, qrange, quantity) for qtype, qranges in realizer.quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities), selection=self.quantifiers)

        self.zero_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.zero_quantifiers))
        self.zero_included_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.zero_included_quantifiers))
        self.zero_negated_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.zero_negated_quantifiers))
        self.all_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.all_quantifiers))
        self.all_included_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.all_included_quantifiers))
        self.all_negated_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.all_negated_quantifiers))
        self.tautological_quantifiers = set(Quantifier.filter(quantifiers=self.quantifiers, selection=Quantifier.tautological_quantifiers))
        self.quantifiers = {qtype_key: [(qrange, quantity) for qtype, qrange, quantity in self.quantifiers if qtype == qtype_key] for qtype_key, _, _ in self.quantifiers}

        return True

    def pn_length(self):
        return self.restrictor_captioner.pn_length() + self.body_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(QuantifierCaptioner, self).pn_symbols() | {'{}-{}-{}-{}'.format(Quantifier.__name__, qtype, *quantifier[:2 - int(qtype == 'composed')]) for qtype, quantifiers in self.quantifiers.items() for quantifier in quantifiers}

    def pn_arity(self):
        arity = super(QuantifierCaptioner, self).pn_arity()
        arity.update({'{}-{}-{}-{}'.format(Quantifier.__name__, qtype, *quantifier[:2 - int(qtype == 'composed')]): 2 for qtype, quantifiers in self.quantifiers.items() for quantifier in quantifiers})
        return arity

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(QuantifierCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        # predication = predication.copy()

        if not self.body_captioner.sample_values(mode=mode, predication=predication):
            return False
        if not self.restrictor_captioner.sample_values(mode=mode, predication=predication):
            return False

        self.qtype = choice(list(self.quantifiers))
        self.qrange, self.quantity = choice(self.quantifiers[self.qtype])

        if (self.qtype, self.qrange, self.quantity) in self.zero_quantifiers:
            assert self.body_captioner.incorrect_possible()
            self.zero_quantification = True
        elif (self.qtype, self.qrange, self.quantity) in self.zero_included_quantifiers:
            self.zero_quantification = self.body_captioner.incorrect_possible() and random() < self.zero_quantification_rate
        else:
            self.zero_quantification = False

        if (self.qtype, self.qrange, self.quantity) in self.all_quantifiers:
            self.all_quantification = True
        elif (self.qtype, self.qrange, self.quantity) in self.all_included_quantifiers:
            self.all_quantification = random() < self.all_quantification_rate
        else:
            self.all_quantification = False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and not self.restrictor_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 1 and not self.body_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 0 and self.zero_quantification:
                continue
            elif self.incorrect_mode in (0, 1) and (self.qtype, self.qrange, self.quantity) in self.tautological_quantifiers:
                # always true in whatever way restrictor/body is changed
                continue
            elif self.incorrect_mode == 3 and not any(q == self.quantity and r != self.qrange for r, q in self.quantifiers[self.qtype]):
                continue
            elif self.incorrect_mode == 4 and not any(r == self.qrange and q != self.quantity for r, q in self.quantifiers[self.qtype]):
                continue
            break
        else:
            return False

        if self.incorrect_mode < 2:
            self.incorrect_qrange = self.qrange
            self.incorrect_quantity = self.quantity

        else:  # incorrect quantifier
            if self.incorrect_mode == 2:
                closest_quantities = list()
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):

                if self.incorrect_mode == 2:  # 2: closest quantity
                    self.incorrect_qrange = self.qrange
                    self.incorrect_quantity = None
                    if self.qrange in ('lt', 'leq') or (self.qrange in ('eq', 'neq') and random() < 0.5):
                        for r, q in self.quantifiers[self.qtype]:
                            if r != self.qrange:
                                continue
                            elif self.qtype == 'ratio' and q >= self.quantity:
                                continue
                            elif self.qtype == 'count' and (q + 1000 * (q < 0) >= self.quantity + 1000 * (self.quantity < 0)):
                                continue
                            elif q in closest_quantities:
                                continue
                            elif self.incorrect_quantity is None or q > self.incorrect_quantity:
                                self.incorrect_quantity = q
                    else:
                        for r, q in self.quantifiers[self.qtype]:
                            if r != self.qrange:
                                continue
                            elif self.qtype == 'ratio' and q <= self.quantity:
                                continue
                            elif self.qtype == 'count' and (q + 1000 * (q < 0) <= self.quantity + 1000 * (self.quantity < 0)):
                                continue
                            elif q in closest_quantities:
                                continue
                            elif self.incorrect_quantity is None or q < self.incorrect_quantity:
                                self.incorrect_quantity = q
                    if self.incorrect_quantity is None:
                        return False
                    closest_quantities.append(self.incorrect_quantity)
                if self.incorrect_mode == 3:  # 3: incorrect range
                    self.incorrect_qrange = choice([r for r, q in self.quantifiers[self.qtype] if q == self.quantity and r != self.qrange])
                    self.incorrect_quantity = self.quantity
                elif self.incorrect_mode == 4:  # 4: incorrect quantity
                    self.incorrect_qrange = self.qrange
                    self.incorrect_quantity = choice([q for r, q in self.quantifiers[self.qtype] if r == self.qrange and q != self.quantity])
                elif self.incorrect_mode == 5:  # 5: incorrect quantifier of same type
                    self.incorrect_qrange, self.incorrect_quantity = choice(self.quantifiers[self.qtype])

                if Quantifier.tautological(qtype=self.qtype, qrange1=self.qrange, quantity1=self.quantity, qrange2=self.incorrect_qrange, quantity2=self.incorrect_quantity):
                    continue
                elif (self.qtype, self.incorrect_qrange, self.incorrect_quantity) in self.tautological_quantifiers:
                    # always true, so never incorrect
                    continue
                elif self.zero_quantification and (self.qtype, self.incorrect_qrange, self.incorrect_quantity) in self.zero_included_quantifiers:
                    # always true if zero, so never incorrect
                    continue
                elif not self.zero_quantification and (self.qtype, self.incorrect_qrange, self.incorrect_quantity) in Quantifier.zero_negated_quantifiers:
                    # always true unless zero, so never incorrect
                    continue
                elif self.all_quantification and (self.qtype, self.incorrect_qrange, self.incorrect_quantity) in self.all_included_quantifiers:
                    # always true if all, so never incorrect
                    continue
                elif not self.all_quantification and (self.qtype, self.incorrect_qrange, self.incorrect_quantity) in self.all_negated_quantifiers:
                    # always true unless all, so never incorrect
                    continue
                break

            else:
                return False

        return True

    def incorrect_possible(self):
        return True

    def model(self):
        model = super(QuantifierCaptioner, self).model()
        model.update(
            qtype=self.qtype,
            qrange=self.qrange,
            quantity=self.quantity,
            incorrect_mode=self.incorrect_mode,
            zero_quantification=self.zero_quantification,
            all_quantification=self.all_quantification,
            restrictor_captioner=self.restrictor_captioner.model(),
            body_captioner=self.body_captioner.model()
        )
        if self.incorrect_mode >= 2:  # incorrect quantifier
            model.update(
                incorrect_qrange=self.incorrect_qrange,
                incorrect_quantity=self.incorrect_quantity
            )
        return model

    def caption(self, predication, world):
        assert predication.empty()

        rstr_body_predication = predication.copy()
        if self.zero_quantification:
            # special case: zero quantifier, hence incorrect body
            body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
            if body is None:
                return None

            restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
            if restrictor is None:
                return None

            body_predication = predication.copy()
            if not self.body_captioner.incorrect(caption=body, predication=body_predication, world=world):
                return None

        else:
            body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
            if body is None:
                return None

            restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
            if restrictor is None:
                return None

        # also for incorrect
        # if not self.pragmatical_tautology and not self.all_quantification and len(rstr_body_predication.agreeing) > 1 and (body_predication.equals(other=rstr_body_predication) or rstr_predication.equals(other=rstr_body_predication)):
        #     # all quantification is inherently tautological
        #     return None

        quantifier = Quantifier(qtype=self.qtype, qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, body=body)

        if not self.correct(caption=quantifier, predication=predication):
            return None

        return quantifier

    def correct(self, caption, predication):
        rstr_predication, _ = caption.apply_to_predication(predication=predication)

        return caption.quantity >= 0 or -(caption.quantity + 1) <= rstr_predication.num_agreeing

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: incorrect restrictor
            rstr_predication = predication.copy()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False

        elif self.incorrect_mode == 1:  # 1: incorrect body
            body_predication = predication.copy()
            # self.restrictor_captioner.correct(caption=caption.restrictor, predication=rstr_predication)
            if self.zero_quantification:
                caption.body = self.body_captioner.caption(predication=body_predication, world=world)
                if caption.body is None:
                    return False
            else:
                if not self.body_captioner.incorrect(caption=caption.body, predication=body_predication, world=world):
                    return False

        elif self.incorrect_mode >= 2:  # incorrect quantifier
            caption.qrange = self.incorrect_qrange
            caption.quantity = self.incorrect_quantity

        # if not self.pragmatical_tautology and not self.all_quantification and rstr_predication.equals(other=body_predication):
        #     # all quantification is inherently tautological
        #     return False

        return self.correct(caption=caption, predication=predication)
