from shapeworld.captions import EntityType
from shapeworld.captioners import WorldCaptioner


class EmptyTypeCaptioner(WorldCaptioner):

    def __init__(
        self,
        pragmatical_redundancy_rate=1.0,
        pragmatical_tautology_rate=0.0,
        logical_redundancy_rate=1.0,
        logical_tautology_rate=1.0,
        logical_contradiction_rate=0.0
    ):
        assert logical_tautology_rate == 1.0
        super(EmptyTypeCaptioner, self).__init__(
            internal_captioners=(),
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )

    def rpn_length(self):
        return 2

    def rpn_symbols(self):
        return super(EmptyTypeCaptioner, self).rpn_symbols() | {'0', EntityType.__name__}

    def incorrect_possible(self):
        return False

    def caption(self, predication, world):
        if predication.num_agreeing == 0:
            return None

        entity_type = EntityType()

        entity_type.apply_to_predication(predication=predication)

        return entity_type

    def incorrect(self, caption, predication, world):
        assert False
