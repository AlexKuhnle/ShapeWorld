from shapeworld.captions import EntityType
from shapeworld.captioners import WorldCaptioner


class EmptyTypeCaptioner(WorldCaptioner):

    def __init__(self):
        super(EmptyTypeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=1.0,
            pragmatical_tautology_rate=1.0,
            logical_redundancy_rate=1.0,
            logical_tautology_rate=1.0,
            logical_contradiction_rate=0.0
        )

    def pn_length(self):
        return 2

    def pn_symbols(self):
        return super(EmptyTypeCaptioner, self).pn_symbols() | {EntityType.__name__ + '0'}

    def pn_arity(self):
        arity = super(EmptyTypeCaptioner, self).pn_arity()
        arity[EntityType.__name__ + '0'] = 0
        return arity

    def incorrect_possible(self):
        return False

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        entity_type = EntityType()

        if not self.correct(caption=entity_type, predication=predication):
            return None

        return entity_type
