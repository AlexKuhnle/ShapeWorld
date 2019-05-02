from random import choice, random
from shapeworld import util
from shapeworld.captions import Quantifier, ComparativeQuantifier
from shapeworld.captioners import WorldCaptioner


class ComparativeQuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect restrictor
    # 1: incorrect comparison
    # 2: incorrect body
    # 3: closest quantity
    # 4: incorrect range
    # 5: incorrect quantity
    # 6: incorrect quantifier of same type

    def __init__(
        self,
        restrictor_captioner,
        comparison_captioner,
        body_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        comparative_quantifiers=None,
        incorrect_distribution=(2, 2, 2, 3, 1, 1, 1)
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
            self.comparative_quantifiers = Quantifier.filter(quantifiers=((qtype, qrange, quantity) for qtype, qranges in realizer.comparative_quantifiers.items() for qrange, quantities in qranges.items() for quantity in quantities), selection=self.comparative_quantifiers)

        self.comparative_quantifiers = {qtype_key: [(qrange, quantity) for qtype, qrange, quantity in self.comparative_quantifiers if qtype == qtype_key] for qtype_key, _, _ in self.comparative_quantifiers}

        return True

    def pn_length(self):
        return self.restrictor_captioner.pn_length() + self.comparison_captioner.pn_length() + self.body_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(ComparativeQuantifierCaptioner, self).pn_symbols() | {'{}-{}-{}-{}'.format(ComparativeQuantifier.__name__, qtype, *quantifier[:2 - int(qtype == 'composed')]) for qtype, quantifiers in self.comparative_quantifiers.items() for quantifier in quantifiers}

    def pn_arity(self):
        arity = super(ComparativeQuantifierCaptioner, self).pn_arity()
        arity.update({'{}-{}-{}-{}'.format(ComparativeQuantifier.__name__, qtype, *quantifier[:2 - int(qtype == 'composed')]): 3 for qtype, quantifiers in self.comparative_quantifiers.items() for quantifier in quantifiers})
        return arity

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(ComparativeQuantifierCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        # predication = predication.copy()

        if not self.body_captioner.sample_values(mode=mode, predication=predication):
            return False
        if not self.restrictor_captioner.sample_values(mode=mode, predication=predication.copy()):
            return False
        if not self.comparison_captioner.sample_values(mode=mode, predication=predication.copy()):
            return False

        # rstr_predication = predication.copy()
        # comp_predication = predication.copy()
        # if not self.restrictor_captioner.sample_values(mode=mode, predication=rstr_predication):
        #     return False
        # if not self.comparison_captioner.sample_values(mode=mode, predication=comp_predication):
        #     return False
        # union_predication = rstr_predication.union(other=comp_predication)
        # if self.body_captioner.sample_values(mode=mode, predication=union_predication):
        #     break

        self.qtype = choice(list(self.comparative_quantifiers))
        self.qrange, self.quantity = choice(self.comparative_quantifiers[self.qtype])

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and not self.restrictor_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 1 and not self.comparison_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 2 and not self.body_captioner.incorrect_possible():
                continue
            elif self.incorrect_mode == 4 and not any(q == self.quantity and r != self.qrange for r, q in self.comparative_quantifiers[self.qtype]):
                continue
            elif self.incorrect_mode == 5 and not any(r == self.qrange and q != self.quantity for r, q in self.comparative_quantifiers[self.qtype]):
                continue
            else:
                break
        else:
            return False

        if self.incorrect_mode >= 3:  # incorrect quantifier
            if self.incorrect_mode == 3:
                closest_quantities = list()
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):

                if self.incorrect_mode == 3:  # 3: closest quantity
                    self.incorrect_qrange = self.qrange
                    self.incorrect_quantity = None
                    if self.qrange in ('lt', 'leq') or (self.qrange in ('eq', 'neq') and random() < 0.5):
                        for r, q in self.comparative_quantifiers[self.qtype]:
                            if r != self.qrange or q >= self.quantity:
                                continue
                            elif q in closest_quantities:
                                continue
                            elif self.incorrect_quantity is None or q > self.incorrect_quantity:
                                self.incorrect_quantity = q
                    else:
                        for r, q in self.comparative_quantifiers[self.qtype]:
                            if r != self.qrange or q <= self.quantity:
                                continue
                            elif q in closest_quantities:
                                continue
                            elif self.incorrect_quantity is None or q < self.incorrect_quantity:
                                self.incorrect_quantity = q
                    if self.incorrect_quantity is None:
                        return False
                    closest_quantities.append(self.incorrect_quantity)
                if self.incorrect_mode == 4:  # 4: incorrect range
                    self.incorrect_qrange = choice([r for r, q in self.comparative_quantifiers[self.qtype] if q == self.quantity and r != self.qrange])
                    self.incorrect_quantity = self.quantity
                elif self.incorrect_mode == 5:  # 5: incorrect quantity
                    self.incorrect_qrange = self.qrange
                    self.incorrect_quantity = choice([q for r, q in self.comparative_quantifiers[self.qtype] if r == self.qrange and q != self.quantity])
                elif self.incorrect_mode == 6:  # 6: incorrect quantifier of same type
                    self.incorrect_qrange, self.incorrect_quantity = choice(self.comparative_quantifiers[self.qtype])

                if Quantifier.tautological(qtype=self.qtype, qrange1=self.qrange, quantity1=self.quantity, qrange2=self.incorrect_qrange, quantity2=self.incorrect_quantity):
                    continue
                break

            else:
                return False

        return True

    def incorrect_possible(self):
        return True

    def model(self):
        model = super(ComparativeQuantifierCaptioner, self).model()
        model.update(
            qtype=self.qtype,
            qrange=self.qrange,
            quantity=self.quantity,
            incorrect_mode=self.incorrect_mode,
            restrictor_captioner=self.restrictor_captioner.model(),
            comparison_captioner=self.comparison_captioner.model(),
            body_captioner=self.body_captioner.model()
        )
        if self.incorrect_mode >= 3:  # incorrect quantifier
            model.update(
                incorrect_qrange=self.incorrect_qrange,
                incorrect_quantity=self.incorrect_quantity
            )
        return model

    def caption(self, predication, world):
        assert predication.empty()

        rstr_body_predication = predication.copy()
        body = self.body_captioner.caption(predication=rstr_body_predication, world=world)
        if body is None:
            return None
        comp_body_predication = rstr_body_predication.copy()

        restrictor = self.restrictor_captioner.caption(predication=rstr_body_predication, world=world)
        if restrictor is None:
            return None

        comparison = self.comparison_captioner.caption(predication=comp_body_predication, world=world)
        if comparison is None:
            return None

        # also for incorrect
        # if not self.pragmatical_tautology and (rstr_predication.equals(other=body_predication) or comp_predication.equals(other=body_predication)):
        #     return None

        quantifier = ComparativeQuantifier(qtype=self.qtype, qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, comparison=comparison, body=body)

        if not self.correct(caption=quantifier, predication=predication):
            return None

        return quantifier

    def correct(self, caption, predication):
        rstr_predication, rstr_body_predication, comp_predication, _, _ = caption.apply_to_predication(predication=predication)

        # restrictor and comparison should not be equal
        return (caption.quantity >= 0 or -caption.quantity <= rstr_body_predication.num_agreeing) and not rstr_predication.equals(other=comp_predication)

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: incorrect restrictor
            rstr_predication = predication.copy()
            if not self.restrictor_captioner.incorrect(caption=caption.restrictor, predication=rstr_predication, world=world):
                return False

        elif self.incorrect_mode == 1:  # 1: incorrect comparison
            comp_predication = predication.copy()
            if not self.comparison_captioner.incorrect(caption=caption.comparison, predication=comp_predication, world=world):
                return False

        elif self.incorrect_mode == 2:  # 2: incorrect body
            rstr_predication = predication.copy()
            self.restrictor_captioner.correct(caption=caption.restrictor, predication=rstr_predication)

            comp_predication = predication.copy()
            self.comparison_captioner.correct(caption=caption.comparison, predication=comp_predication)

            body_predication = rstr_predication.union(other=comp_predication)
            if not self.body_captioner.incorrect(caption=caption.body, predication=body_predication, world=world):
                return False

        elif self.incorrect_mode >= 3:  # incorrect quantifier
            caption.qrange = self.incorrect_qrange
            caption.quantity = self.incorrect_quantity

        # if not self.pragmatical_tautology and (rstr_predication.equals(other=body_predication) or comp_predication.equals(other=body_predication)):
        #     return False

        return self.correct(caption=caption, predication=predication)
