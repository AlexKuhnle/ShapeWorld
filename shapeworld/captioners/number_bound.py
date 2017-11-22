from random import choice, random
from shapeworld import util
from shapeworld.captions import NumberBound
from shapeworld.captioners import WorldCaptioner


class NumberBoundCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect quantifier
    # 2: number off by one
    # 3: number off by two
    # 4: random incorrect number

    def __init__(self, quantifier_captioner, number_bounds=None, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
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
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [3, 1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(NumberBoundCaptioner, self).set_realizer(realizer):
            return False

        if self.number_bounds is None:
            self.number_bounds = list(realizer.number_bounds)
        else:
            self.number_bounds = [bound for bound in realizer.number_bounds if bound in self.number_bounds]

        return True

    def rpn_length(self):
        return self.quantifier_captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(NumberBoundCaptioner, self).rpn_symbols() | {'{}-{}'.format(NumberBound.__name__, bound) for bound in self.number_bounds}

    def sample_values(self, mode, correct, predication):
        assert predication.empty()

        if not super(NumberBoundCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        if not self.quantifier_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1), predication=predication):  # 1: incorrect quantifier
            return False

        # potentially option to choose fixed number bound?
        # ----->
        # qtype = self.quantifier_captioner.qtype
        # quantity = self.quantifier_captioner.quantity
        # for _ in range(self.__class__.MAX_SAMPLE_ATTEMPTS):
        #     self.bound = choice(self.number_bounds)
        #     if not correct or ((qtype != 'count' or self.bound >= quantity) and (qtype != 'ratio' or quantity == 0.0 or self.bound >= 1.0 // quantity)):
        #         break
        # else:
        #     return False

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(NumberBoundCaptioner, self).model(),
            dict2=dict(
                incorrect_mode=self.incorrect_mode,
                quantifier_captioner=self.quantifier_captioner.model()
            )
        )

    def caption(self, predication, world):
        assert predication.empty()

        quant_predication = predication.sub_predication()
        quantifier = self.quantifier_captioner.caption(predication=quant_predication, world=world)
        if quantifier is None:
            return None

        num_predication = predication.sub_predication()
        self.quantifier_captioner.restrictor_captioner.apply_caption_to_predication(caption=quantifier.restrictor, predication=num_predication)

        if num_predication.num_agreeing not in self.number_bounds and (self.incorrect_mode == 0 or self.incorrect_mode == 1):
            return None

        else:
            # potentially option to choose fixed number bound?
            return NumberBound(bound=num_predication.num_agreeing, quantifier=quantifier)

    def incorrect(self, caption, predication, world):
        assert predication.empty()

        if self.incorrect_mode == 0:  # 0: correct
            self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 1:  # 1: incorrect quantifier
            quant_predication = predication.sub_predication()
            if not self.quantifier_captioner.incorrect(caption=caption.quantifier, predication=quant_predication, world=world):
                return False
            num_predication = predication.sub_predication()
            self.quantifier_captioner.restrictor_captioner.apply_caption_to_predication(caption=caption.quantifier.restrictor, predication=num_predication)

        elif self.incorrect_mode == 2 or self.incorrect_mode == 3:  # 2/3: number off by one/two
            num_predication = self.apply_caption_to_predication(caption=caption, predication=predication)

            delta = self.incorrect_mode - 1
            if random() < 0.5:
                caption.bound = num_predication.num_agreeing - delta
                if caption.bound not in self.number_bounds:
                    caption.bound = num_predication.num_not_disagreeing + delta
            else:
                caption.bound = num_predication.num_not_disagreeing + delta
                if caption.bound not in self.number_bounds:
                    caption.bound = num_predication.num_not_disagreeing - delta
            if caption.bound not in self.number_bounds:
                return False

        elif self.incorrect_mode == 4:  # 4: random incorrect number
            self.apply_caption_to_predication(caption=caption, predication=predication)
            caption.bound = choice(self.number_bounds)

        return True

    def apply_caption_to_predication(self, caption, predication):
        quant_predication = predication.sub_predication()
        self.quantifier_captioner.apply_caption_to_predication(caption=caption.quantifier, predication=quant_predication)
        num_predication = predication.sub_predication()
        self.quantifier_captioner.restrictor_captioner.apply_caption_to_predication(caption=caption.quantifier.restrictor, predication=num_predication)
        return num_predication
