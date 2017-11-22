from shapeworld.captions import Predicate


class EntityType(Predicate):

    predtypes = {'type'}

    def __init__(self, predicates=None):
        if predicates is None:
            predicates = dict()
        elif isinstance(predicates, Predicate):
            predicates = {predicates.predtype: predicates}
        assert isinstance(predicates, dict)
        assert all(isinstance(predicate, Predicate) and not isinstance(predicate, EntityType) for predicate in predicates.values())
        super(EntityType, self).__init__(predtype='type', value=predicates)

    def model(self):
        return dict(
            component=str(self),
            predtype=self.predtype,
            value={predtype: predicate.model() for predtype, predicate in self.value.items()}
        )

    def reverse_polish_notation(self):
        return [rpn_symbol for predtype in sorted(self.value) for rpn_symbol in self.value[predtype].reverse_polish_notation()] + \
            [str(len(self.value)), str(self)]  # two separate arguments, no tuple?

    def pred_agreement(self, entity, predication):
        return all(predicate.pred_agreement(entity=entity, predication=predication) for predicate in self.value.values())

    def pred_disagreement(self, entity, predication):
        return any(predicate.pred_disagreement(entity=entity, predication=predication) for predicate in self.value.values())
