from shapeworld import util
from shapeworld.captions import Attribute, EntityType, Relation
from shapeworld.captioners import WorldCaptioner


class AttributeTypeRelationCaptioner(WorldCaptioner):

    def __init__(
        self,
        attribute_type_captioner,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=0.0,
        logical_tautology_rate=0.0,
        logical_contradiction_rate=0.0
    ):
        super(AttributeTypeRelationCaptioner, self).__init__(
            internal_captioners=(attribute_type_captioner,),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

        self.attribute_type_captioner = attribute_type_captioner

    def pn_length(self):
        return self.attribute_type_captioner.pn_length() + 1

    def pn_symbols(self):
        return super(AttributeTypeRelationCaptioner, self).pn_symbols() | \
            {'{}-{}'.format(Relation.__name__, 'attribute'), '{}-{}'.format(Relation.__name__, 'type')}

    def pn_arity(self):
        arity = super(AttributeTypeRelationCaptioner, self).pn_arity()
        arity['{}-{}'.format(Relation.__name__, 'attribute')] = 1
        arity['{}-{}'.format(Relation.__name__, 'type')] = 1
        return arity

    def sample_values(self, mode, predication):
        if not super(AttributeTypeRelationCaptioner, self).sample_values(mode=mode, predication=predication):
            return False

        return self.attribute_type_captioner.sample_values(mode=mode, predication=predication)

    def model(self):
        return util.merge_dicts(
            dict1=super(AttributeTypeRelationCaptioner, self).model(),
            dict2=dict(
                attribute_type_captioner=self.attribute_type_captioner.model()
            )
        )

    def incorrect_possible(self):
        return self.attribute_type_captioner.incorrect_possible()

    def caption(self, predication, world):
        caption = self.attribute_type_captioner.caption(predication=predication, world=world)
        if caption is None:
            return None

        elif isinstance(caption, Attribute):
            return Relation(predtype='attribute', value=caption)

        elif isinstance(caption, EntityType):
            return Relation(predtype='type', value=caption)

        else:
            raise NotImplementedError

    def incorrect(self, caption, predication, world):
        return self.attribute_type_captioner.incorrect(caption=caption.value, predication=predication, world=world)
