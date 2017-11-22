from shapeworld import util
from shapeworld.captions import Attribute, EntityType, Relation
from shapeworld.captioners import WorldCaptioner


class AttributeTypeRelationCaptioner(WorldCaptioner):

    def __init__(self, attribute_type_captioner=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        super(AttributeTypeRelationCaptioner, self).__init__(
            internal_captioners=(attribute_type_captioner,),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.captioner = attribute_type_captioner

    def rpn_length(self):
        return self.captioner.rpn_length() + 1

    def rpn_symbols(self):
        return super(AttributeTypeRelationCaptioner, self).rpn_symbols() | \
            {'{}-{}'.format(Relation.__name__, 'attribute'), '{}-{}'.format(Relation.__name__, 'type'), EntityType.__name__}

    def sample_values(self, mode, correct, predication):
        if not super(AttributeTypeRelationCaptioner, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        return self.captioner.sample_values(mode=mode, correct=correct, predication=predication)

    def model(self):
        return util.merge_dicts(
            dict1=super(AttributeTypeRelationCaptioner, self).model(),
            dict2=dict(
                captioner=self.captioner.model()
            )
        )

    def caption(self, predication, world):
        caption = self.captioner.caption(predication=predication, world=world)
        if caption is None:
            return None

        elif isinstance(caption, Attribute):
            return Relation(predtype='attribute', value=caption)

        elif isinstance(caption, EntityType):
            return Relation(predtype='type', value=caption)

        else:
            assert False

    def incorrect(self, caption, predication, world):
        return self.captioner.incorrect(caption=caption.value, predication=predication, world=world)

    def apply_caption_to_predication(self, caption, predication):
        self.captioner.apply_caption_to_predication(caption=caption.value, predication=predication)
