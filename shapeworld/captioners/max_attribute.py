from random import choice
from shapeworld import util
from shapeworld.captions import Attribute
from shapeworld.captioners import WorldCaptioner


class MaxAttributeCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect scope
    # 2: incorrect max attribute
    # 3: inverse max attribute

    def __init__(self, scope_captioner, attributes=None, incorrect_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(MaxAttributeCaptioner, self).__init__(
            internal_captioners=(scope_captioner,),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.scope_captioner = scope_captioner
        self.attributes = attributes
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(MaxAttributeCaptioner, self).set_realizer(realizer):
            return False

        if self.attributes is None:
            self.attributes = list((predtype, value) for predtype, values in realizer.attributes.items() if predtype[-4:] == '-max' for value in values)
        else:
            self.attributes = list((predtype, value) for predtype, values in realizer.attributes.items() if predtype in self.attributes for value in values)

        return True

    def rpn_length(self):
        return super(MaxAttributeCaptioner, self).rpn_length() + 1

    def rpn_symbols(self):
        return super(MaxAttributeCaptioner, self).rpn_symbols() | {'{}-{}-{}'.format(Attribute.__name__, *attribute) for attribute in self.attributes}

    def sample_values(self, mode, correct, predication):
        if not super(MaxAttributeCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        self.predtype, self.value = choice(self.attributes)

        if not self.scope_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1), predication=predication):  # 1: incorrect scope
            return False

        # instead of self.logical_redundancy, since it uniquely selects one entity
        if not self.logical_tautology and predication.redundant(predicate=self.predtype):
            return False

        if self.incorrect_mode == 2:  # 2: incorrect max attribute
            self.incorrect_attributes = [(predtype, value) for predtype, value in self.attributes if predtype != self.predtype or value != self.value]

        predication.apply(predicate=self.predtype)

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(MaxAttributeCaptioner, self).model(),
            dict2=dict(
                predtype=self.predtype,
                value=self.value,
                incorrect_mode=self.incorrect_mode,
                scope_captioner=self.scope_captioner.model()
            )
        )

    def caption(self, predication, world):
        scope = self.scope_captioner.caption(predication=predication, world=world)
        if scope is None:
            return None

        max_attribute = Attribute(predtype=self.predtype, value=self.value)

        scope_predication_copy = predication.copy(reset=True)
        self.scope_captioner.apply_caption_to_predication(caption=scope, predication=scope_predication_copy)

        if predication.contradictory(predicate=max_attribute, predication=scope_predication_copy):
            return None
        elif not self.pragmatical_redundancy and predication.redundant(predicate=max_attribute, predication=scope_predication_copy):
            return None

        predication.apply(predicate=max_attribute, predication=scope_predication_copy)
        scope.value[self.predtype] = max_attribute

        return scope

    def incorrect(self, caption, predication, world):
        if self.incorrect_mode == 0:  # 0: correct
            self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 1:  # 1: incorrect scope
            if not self.scope_captioner.incorrect(caption=caption, predication=predication, world=world):
                return False
            scope_predication_copy = predication.copy(reset=True)
            self.scope_captioner.apply_caption_to_predication(caption=caption, predication=scope_predication_copy)
            predication.apply(predicate=caption.value[self.predtype], predication=scope_predication_copy)

        elif self.incorrect_mode == 2:  # 2: incorrect max attribute
            max_attribute = caption.value.pop(self.predtype)
            max_attribute.predtype, max_attribute.value = choice(self.incorrect_attributes)
            self.predtype = max_attribute.predtype
            caption.value[self.predtype] = max_attribute
            self.apply_caption_to_predication(caption=caption, predication=predication)

        elif self.incorrect_mode == 3:  # 3: inverse max attribute
            max_attribute = caption.value[self.predtype]
            max_attribute.value = -max_attribute.value
            if (max_attribute.predtype, max_attribute.value) not in self.attributes:
                return False
            self.apply_caption_to_predication(caption=caption, predication=predication)

        return True

    def apply_caption_to_predication(self, caption, predication):
        self.scope_captioner.apply_caption_to_predication(caption=caption, predication=predication)
        scope_predication_copy = predication.copy(reset=True)
        self.scope_captioner.apply_caption_to_predication(caption=caption, predication=scope_predication_copy)
        predication.apply(predicate=caption.value[self.predtype], predication=scope_predication_copy)
