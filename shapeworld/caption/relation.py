from shapeworld.caption import Predicate


class Relation(Predicate):

    # can be unary, binary, binary with generalized quantification
    # dmrs might need to insert quantifier

    __slots__ = ('reltype', 'reference')

    def __init__(self, reltype, reference):
        assert reltype in ('left', 'right', 'above', 'below')
        assert isinstance(reference, Predicate)
        self.reltype = reltype
        self.reference = reference

    def model(self):
        return {'component': 'relation', 'reltype': self.reltype, 'reference': self.reference.model()}

    def agreeing_entities(self, entities):
        reference_entities = self.reference.agreeing_entities(entities=entities)

        if self.reltype == 'left':
            return [entity for entity in entities if any(reference['center']['x'] - entity['center']['x'] > abs(entity['center']['y'] - reference['center']['y']) for reference in reference_entities)]

        elif self.reltype == 'right':
            return [entity for entity in entities if any(entity['center']['x'] - reference['center']['x'] > abs(entity['center']['y'] - reference['center']['y']) for reference in reference_entities)]

        elif self.reltype == 'above':
            return [entity for entity in entities if any(reference['center']['y'] - entity['center']['y'] > abs(entity['center']['x'] - reference['center']['x']) for reference in reference_entities)]

        elif self.reltype == 'below':
            return [entity for entity in entities if any(entity['center']['y'] - reference['center']['y'] > abs(entity['center']['x'] - reference['center']['x']) for reference in reference_entities)]

    def disagreeing_entities(self, entities):
        reference_entities = self.reference.agreeing_entities(entities=entities)

        if self.reltype == 'left':
            return [entity for entity in entities if all(reference['center']['x'] - entity['center']['x'] <= abs(entity['center']['y'] - reference['center']['y']) for reference in reference_entities)]

        elif self.reltype == 'right':
            return [entity for entity in entities if all(entity['center']['x'] - reference['center']['x'] <= abs(entity['center']['y'] - reference['center']['y']) for reference in reference_entities)]

        elif self.reltype == 'above':
            return [entity for entity in entities if all(reference['center']['y'] - entity['center']['y'] <= abs(entity['center']['x'] - reference['center']['x']) for reference in reference_entities)]

        elif self.reltype == 'below':
            return [entity for entity in entities if all(entity['center']['y'] - reference['center']['y'] <= abs(entity['center']['x'] - reference['center']['x']) for reference in reference_entities)]
