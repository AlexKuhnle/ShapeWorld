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
        comparison_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        selectors=None,
        incorrect_distribution=(1, 1)
    ):
        super(SelectorCaptioner, self).__init__(
            internal_captioners=(scope_captioner, comparison_captioner),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.scope_captioner = scope_captioner
        self.comparison_captioner = comparison_captioner
        self.selectors = selectors
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(SelectorCaptioner, self).set_realizer(realizer):
            return False

        if self.selectors is None:
            self.selectors = [(predtype, value) for predtype, values in realizer.selectors.items() for value in values]
        else:
            self.selectors = [
                (predtype, value) for predtype, values in realizer.selectors.items() for value in values
                if any((p == '*' or predtype == p) and (v == '*' or value == v) for p, v in self.selectors)
            ]

        return True

    def pn_length(self):
        return self.scope_captioner.pn_length() + self.comparison_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(SelectorCaptioner, self).pn_symbols() | {'{}-{}-{}'.format(Selector.__name__, *selector) for selector in self.selectors}

    def pn_arity(self):
        arity = super(SelectorCaptioner, self).pn_arity()
        arity.update({
            '{}-{}-{}'.format(Selector.__name__, *selector): 2 if selector[0] in Selector.comparison_selectors else 1 for selector in self.selectors
        })
        return arity

    def sample_values(self, mode, predication):
        if not super(SelectorCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.predtype, self.value = choice(self.selectors)
            if self.predtype in ('size-two', 'size-max') and not self.logical_contradiction and predication.blocked(predicate='shape'):
                continue
            elif self.predtype in ('shade-two', 'shade-max') and not self.logical_contradiction and predication.blocked(predicate='color'):
                continue
            break

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            scope_predication = predication.copy()
            if not self.scope_captioner.sample_values(mode=mode, predication=scope_predication):
                continue
            elif self.predtype in ('size-two', 'size-max') and ((not predication.redundant(predicate='shape') and not scope_predication.redundant(predicate='shape')) or (not self.logical_redundancy and predication.redundant(predicate='shape') and scope_predication.redundant(predicate='shape'))):
                # or (not self.logical_contradiction and scope_predication.blocked(predicate='shape'))
                continue
            elif self.predtype in ('shade-two', 'shade-max') and ((not predication.redundant(predicate='color') and not scope_predication.redundant(predicate='color')) or (not self.logical_redundancy and predication.redundant(predicate='color') and scope_predication.redundant(predicate='color'))):
                # or (not self.logical_contradiction and scope_predication.blocked(predicate='color'))
                continue
            predication.predicates.update(scope_predication.predicates)
            predication.blocked_preds.update(scope_predication.blocked_preds)
            break
        else:
            return False

        self.incorrect_mode = util.sample(self.incorrect_distribution)

        self.incorrect_predtype = self.predtype
        self.incorrect_value = self.value

        if self.incorrect_mode == 0:  # 0: incorrect selectors
            for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
                self.incorrect_predtype, self.incorrect_value = choice(self.selectors)
                if self.incorrect_predtype == self.predtype and self.incorrect_value == self.value:
                    continue
                elif self.incorrect_predtype in ('size-two', 'size-max') and not self.logical_contradiction and (predication.blocked(predicate='shape') or scope_predication.blocked(predicate='shape')):
                    continue
                elif self.incorrect_predtype in ('shade-two', 'shade-max') and not self.logical_contradiction and (predication.blocked(predicate='color') or scope_predication.blocked(predicate='color')):
                    continue
                break
            else:
                return False

        # if not self.scope_captioner.sample_values(mode=mode, predication=predication):
        #     return False

        if self.predtype in Selector.comparison_selectors or self.incorrect_predtype in Selector.comparison_selectors:
            comp_predication = predication.copy(reset=True)
            if not self.comparison_captioner.sample_values(mode=mode, predication=comp_predication):
                return False

        if self.predtype in ('size-two', 'size-max') or self.incorrect_predtype in ('size-two', 'size-max'):
            predication.apply(predicate='shape')
            if not self.logical_redundancy or (not self.logical_contradiction and scope_predication.redundant(predicate='shape') and self.incorrect_predtype in ('size-two', 'size-max')):
                predication.block(predicate='shape')
        elif self.predtype in ('shade-two', 'shade-max') or self.incorrect_predtype in ('shade-two', 'shade-max'):
            predication.apply(predicate='color')
            if not self.logical_redundancy or (not self.logical_contradiction and scope_predication.redundant(predicate='color') and self.incorrect_predtype in ('shade-two', 'shade-max')):
                predication.block(predicate='color')

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
        if self.predtype in Selector.comparison_selectors or self.incorrect_predtype in Selector.comparison_selectors:
            model.update(
                comparison_captioner=self.comparison_captioner.model()
            )
        return model

    def caption(self, predication, world):
        if self.predtype in Selector.comparison_selectors or self.incorrect_predtype in Selector.comparison_selectors:
            comp_predication = predication.copy(reset=True)
            comparison = self.comparison_captioner.caption(predication=comp_predication, world=world)
            if comparison is None:
                return None
        else:
            comp_predication = None
            comparison = None

        scope_predication = predication.copy()
        scope = self.scope_captioner.caption(predication=scope_predication, world=world)
        if scope is None:
            return None

        selector = Selector(predtype=self.predtype, value=self.value, scope=scope, comparison=comparison)

        if not self.correct(caption=selector, predication=predication):
            return None

        return selector

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: incorrect selector
            caption.predtype = self.incorrect_predtype
            caption.value = self.incorrect_value

        elif self.incorrect_mode == 1:  # 1: inverse selector
            caption.value = -caption.value
            if (caption.predtype, caption.value) not in self.selectors:
                return False

        return self.correct(caption=caption, predication=predication)
