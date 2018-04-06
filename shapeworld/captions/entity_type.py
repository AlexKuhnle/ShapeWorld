from shapeworld.captions import Predicate, Attribute


class EntityType(Predicate):

    predtypes = {'type'}

    def __init__(self, attributes=None):
        if attributes is None:
            attributes = list()
        elif isinstance(attributes, Attribute):
            attributes = [attributes]
        assert isinstance(attributes, list)
        assert all(isinstance(attribute, Attribute) for attribute in attributes)
        super(EntityType, self).__init__(predtype='type', value=attributes)

    def model(self):
        return dict(
            component=str(self),
            predtype=self.predtype,
            value=[attribute.model() for attribute in self.value]
        )

    def reverse_polish_notation(self):
        return [rpn_symbol for attribute in self.value for rpn_symbol in attribute.reverse_polish_notation()] + \
            [str(len(self.value)), str(self)]

    def apply_to_predication(self, predication):
        for attribute in self.value:
            attribute.apply_to_predication(predication=predication)

    def pred_agreement(self, entity):
        return all(attribute.pred_agreement(entity=entity) for attribute in self.value)

    def pred_disagreement(self, entity):
        return any(attribute.pred_disagreement(entity=entity) for attribute in self.value)
