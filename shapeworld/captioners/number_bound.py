from random import choice, random
from shapeworld import util
from shapeworld.captions import NumberBound
from shapeworld.captioners import WorldCaptioner


class NumberBoundCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: incorrect quantifier
    # 1: number off by one
    # 2: number off by two
    # 3: random incorrect number

    def __init__(
        self,
        quantifier_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0,
        number_bounds=None,
        incorrect_distribution=(3, 1, 1, 1)
    ):
        super(NumberBoundCaptioner, self).__init__(
            internal_captioners=(quantifier_captioner,),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.quantifier_captioner = quantifier_captioner
        self.number_bounds = number_bounds
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution)

    def set_realizer(self, realizer):
        if not super(NumberBoundCaptioner, self).set_realizer(realizer):
            return False

        if self.number_bounds is None:
            self.number_bounds = list(realizer.number_bounds)
        else:
            self.number_bounds = [bound for bound in realizer.number_bounds if bound in self.number_bounds]

        return True

    def pn_length(self):
        return self.quantifier_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(NumberBoundCaptioner, self).pn_symbols() | {'{}-{}'.format(NumberBound.__name__, bound) for bound in self.number_bounds}

    def pn_arity(self):
        arity = super(NumberBoundCaptioner, self).pn_arity()
        arity.update({'{}-{}'.format(NumberBound.__name__, bound): 1 for bound in self.number_bounds})
        return arity

    def sample_values(self, mode, predication):
        assert predication.empty()

        if not super(NumberBoundCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        if not self.quantifier_captioner.sample_values(mode=mode, predication=predication):
            return False

        for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
            self.incorrect_mode = util.sample(self.incorrect_distribution)
            if self.incorrect_mode == 0 and not self.quantifier_captioner.incorrect_possible():
                continue
            break
        else:
            return False

        # potentially option to choose fixed number bound?
        # self.bound = choice(self.number_bounds)

        return True

    def incorrect_possible(self):
        return True

    def model(self):
        model = super(NumberBoundCaptioner, self).model()
        model.update(
            incorrect_mode=self.incorrect_mode,
            quantifier_captioner=self.quantifier_captioner.model()
        )
        return model

    def caption(self, predication, world):
        assert predication.empty()

        quant_predication = predication.copy()
        quantifier = self.quantifier_captioner.caption(predication=quant_predication, world=world)
        if quantifier is None:
            return None

        num_predication = predication.copy()
        self.quantifier_captioner.restrictor_captioner.correct(caption=quantifier.restrictor, predication=num_predication)

        # potentially option to choose fixed number bound?
        number_bound = NumberBound(bound=num_predication.num_agreeing, quantifier=quantifier)

        if not self.correct(caption=number_bound, predication=predication):
            return None

        return number_bound

    def correct(self, caption, predication):
        num_predication = caption.apply_to_predication(predication=predication)

        return num_predication.num_agreeing in self.number_bounds

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: incorrect quantifier
            quant_predication = predication.copy()
            if not self.quantifier_captioner.incorrect(caption=caption.quantifier, predication=quant_predication, world=world):
                return False

        elif self.incorrect_mode == 1 or self.incorrect_mode == 2:  # 1/2: number off by one/two
            num_predication = caption.apply_to_predication(predication=predication)

            delta = self.incorrect_mode - 1
            if random() < 0.5:
                caption.bound = num_predication.num_agreeing - delta
                if caption.bound not in self.number_bounds:
                    caption.bound = num_predication.num_not_disagreeing + delta
            else:
                caption.bound = num_predication.num_not_disagreeing + delta
                if caption.bound not in self.number_bounds:
                    caption.bound = num_predication.num_not_disagreeing - delta

            # calls apply_to_predication
            return caption.bound in self.number_bounds

        elif self.incorrect_mode == 3:  # 3: random incorrect number
            caption.bound = choice(self.number_bounds)

        return self.correct(caption=caption, predication=predication)
