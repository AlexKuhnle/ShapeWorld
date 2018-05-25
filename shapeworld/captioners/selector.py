from random import choice
from shapeworld import util
from shapeworld.captions import Selector
from shapeworld.captioners import WorldCaptioner


class SelectorCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect selector
    # 1: inverse selector

    def __init__(
        self,
        scope_captioner,
        reference_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        selectors=None,
        incorrect_distribution=(1, 1)
    ):
        super(SelectorCaptioner, self).__init__(
            internal_captioners=(scope_captioner, reference_captioner),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.scope_captioner = scope_captioner
        self.reference_captioner = reference_captioner
        self.selectors = selectors
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(SelectorCaptioner, self).set_realizer(realizer):
            return False

        if self.selectors is None:
            self.selectors = [(predtype, value) for predtype, values in realizer.selectors.items() for value in values]
        else:
            assert len(self.selectors) == 2
            self.selectors = [
                (predtype, value) for predtype, values in realizer.selectors.items() for value in values
                if any((p == '*' or predtype == p) and (v == '*' or value == v) for p, v in self.selectors)
            ]

        return True

    def rpn_length(self):
        return super(SelectorCaptioner, self).rpn_length() + 1

    def rpn_symbols(self):
        return super(SelectorCaptioner, self).rpn_symbols() | {'{}-{}-{}'.format(Selector.__name__, *selector) for selector in self.selectors}

    def sample_values(self, mode, predication):
        if not super(SelectorCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        if not self.scope_captioner.sample_values(mode=mode, predication=predication):
            return False

        self.predtype, self.value = choice(self.selectors)

        self.incorrect_mode = util.sample(self.incorrect_distribution)

        self.incorrect_predtype = self.predtype
        self.incorrect_value = self.value

        if self.incorrect_mode == 0:  # 0: incorrect selectors
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
                self.incorrect_predtype, self.incorrect_value = choice(self.selectors)
                if self.incorrect_predtype == self.predtype and self.incorrect_value == self.value:
                    continue
                break
            else:
                return False

        if self.predtype in Selector.reference_selectors or self.incorrect_predtype in Selector.reference_selectors:
            ref_predication = predication.copy(reset=True)
            if not self.reference_captioner.sample_values(mode=mode, predication=ref_predication):
                return False

        if self.predtype in ('size-two', 'size-max') or self.incorrect_predtype in ('size-two', 'size-max'):
            predication.apply(predicate='shape')
        elif self.predtype in ('shade-two', 'shade-max') or self.incorrect_predtype in ('shade-two', 'shade-max'):
            predication.apply(predicate='color')

        return True

    def incorrect_possible(self):
        return True

    def model(self):
        model = super(SelectorCaptioner, self).model()
        model.update(
            predtype=self.predtype,
            value=self.value,
            incorrect_mode=self.incorrect_mode,
            scope_captioner=self.scope_captioner.model()
        )
        if self.incorrect_mode == 0:  # 0: incorrect selector
            model.update(
                incorrect_predtype=self.incorrect_predtype,
                incorrect_value=self.incorrect_value
            )
        if self.predtype in Selector.reference_selectors or self.incorrect_predtype in Selector.reference_selectors:
            model.update(
                reference_captioner=self.reference_captioner.model()
            )
        return model

    def caption(self, predication, world):
        scope_predication = predication.sub_predication(reset=True)

        scope = self.scope_captioner.caption(predication=predication, world=world)
        if scope is None:
            return None
        scope.apply_to_predication(predication=scope_predication)

        if self.predtype in Selector.reference_selectors or self.incorrect_predtype in Selector.reference_selectors:
            ref_predication = predication.sub_predication(reset=True)
            reference = self.reference_captioner.caption(predication=ref_predication, world=world)
            if reference is None:
                return None
            if not ref_predication.disjoint(other=scope_predication):
                return None
        else:
            ref_predication = None
            reference = None

        selector = Selector(predtype=self.predtype, value=self.value, scope=scope, reference=reference)

        predication.apply(predicate=selector, scope_predication=scope_predication, ref_predication=ref_predication)

        return selector

    def incorrect(self, caption, predication, world):
        scope_predication = predication.sub_predication(reset=True)
        caption.scope.apply_to_predication(predication=predication)
        caption.scope.apply_to_predication(predication=scope_predication)

        if self.predtype in Selector.reference_selectors or self.incorrect_predtype in Selector.reference_selectors:
            ref_predication = predication.sub_predication(reset=True)
            caption.reference.apply_to_predication(predication=ref_predication)
        else:
            ref_predication = None

        if self.incorrect_mode == 0:  # 0: incorrect selector
            caption.predtype = self.incorrect_predtype
            caption.value = self.incorrect_value

        elif self.incorrect_mode == 1:  # 1: inverse selector
            caption.value = -caption.value
            if (caption.predtype, caption.value) not in self.selectors:
                return False

        else:
            assert False

        predication.apply(predicate=caption, scope_predication=scope_predication, ref_predication=ref_predication)

        return True
