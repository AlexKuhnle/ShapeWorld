from shapeworld.util import Point
from shapeworld.world import Entity, Shape
from shapeworld.caption import Predicate


class Relation(Predicate):

    # can be unary, binary, binary with generalized quantification
    # dmrs might need to insert quantifier

    ternary_relations = ('proximity-rel',)

    __slots__ = ('reltype', 'value', 'reference', 'comparison')

    def __init__(self, reltype, value, reference, comparison=None):
        assert reltype in ('x-rel', 'y-rel', 'z-rel', 'proximity-max', 'proximity-rel', 'size-rel', 'shade-rel')
        assert isinstance(reference, Predicate)
        assert (comparison is None) != (reltype in Relation.ternary_relations)
        assert reltype not in Relation.ternary_relations or isinstance(comparison, Predicate)
        if reltype in ('x-rel', 'y-rel', 'z-rel', 'proximity-max', 'proximity-rel', 'size-rel', 'shade-rel'):
            assert value == -1 or value == 1
        self.reltype = reltype
        self.value = value
        self.reference = reference
        self.comparison = comparison

    def model(self):
        if self.comparison is None:
            return dict(component='relation', reltype=self.reltype, value=self.value, reference=self.reference.model())
        else:
            return dict(component='relation', reltype=self.reltype, value=self.value, reference=self.reference.model(), comparison=self.comparison.model())

    def agreeing_entities(self, entities):
        reference_entities = self.reference.agreeing_entities(entities=entities)

        if self.reltype == 'x-rel':
            # min distance in case of overlap
            return [entity for entity in entities if any((entity['center']['x'] - reference['center']['x']) * self.value > abs(entity['center']['y'] - reference['center']['y']) for reference in reference_entities)]

        elif self.reltype == 'y-rel':
            return [entity for entity in entities if any((entity['center']['y'] - reference['center']['y']) * self.value > abs(entity['center']['x'] - reference['center']['x']) for reference in reference_entities)]

        elif self.reltype == 'z-rel':
            reference_entities = [Entity.from_model(reference) for reference in reference_entities]
            return [entity for entity in entities if any(Entity.from_model(entity).collides(reference) and entity['id'] * self.value > reference.id * self.value for reference in reference_entities)]

        elif self.reltype == 'proximity-max':
            # min distance to second
            entity_centers = [Point.from_model(entity['center']) for entity in entities]
            agreeing_ids = set()
            for reference in reference_entities:
                reference_center = Point.from_model(reference['center'])
                max_distance = -1.0
                max_entity = None
                for entity, entity_center in zip(entities, entity_centers):
                    if entity['id'] == reference['id']:
                        continue
                    distance = (entity_center - reference_center).length() * self.value
                    if distance > max_distance:
                        max_distance = distance
                        max_entity = entity
                if max_entity is not None:
                    agreeing_ids.add(max_entity['id'])
            return [entity for entity in entities if entity['id'] in agreeing_ids]

        elif self.reltype == 'proximity-rel':
            # min distance to second
            comparison_entities = self.comparison.agreeing_entities(entities=entities)
            entity_centers = [Point.from_model(entity['center']) for entity in entities]
            reference_centers = [Point.from_model(reference['center']) for reference in reference_entities]
            comparison_centers = [Point.from_model(comparison['center']) for comparison in comparison_entities]
            agreeing_ids = set()
            for reference, reference_center, comparison, comparison_center in zip(reference_entities, reference_centers, comparison_entities, comparison_centers):
                if reference['id'] == comparison['id']:
                    continue
                reference_distance = (comparison_center - reference_center).length() * self.value
                for entity, entity_center in zip(entities, entity_centers):
                    if entity['id'] == reference['id'] or entity['id'] == comparison['id']:
                        continue
                    if (entity_center - reference_center).length() * self.value > reference_distance:
                        agreeing_ids.add(entity['id'])
            return [entity for entity in entities if entity['id'] in agreeing_ids]

        elif self.reltype == 'size-rel':
            # min difference
            reference_areas = [Shape.from_model(reference['shape']).area() for reference in reference_entities]
            return [entity for entity in entities if any(Shape.from_model(entity['shape']).area() * self.value > reference_area * self.value for reference_area in reference_areas)]

        elif self.reltype == 'shade-rel':
            # min difference
            return [entity for entity in entities if any(entity['color']['shade'] * self.value > reference['color']['shade'] * self.value for reference in reference_entities)]

    def disagreeing_entities(self, entities):
        reference_entities = self.reference.agreeing_entities(entities=entities)

        if self.reltype == 'x-rel':
            return [entity for entity in entities if all((entity['center']['x'] - reference['center']['x']) * self.value < 0 for reference in reference_entities)]

        elif self.reltype == 'y-rel':
            return [entity for entity in entities if all((entity['center']['y'] - reference['center']['y']) * self.value < 0 for reference in reference_entities)]

        elif self.reltype == 'z-rel':
            reference_entities = [Entity.from_model(reference) for reference in reference_entities]
            return [entity for entity in entities if any(Entity.from_model(entity).collides(reference) for reference in reference_entities) and all(not Entity.from_model(entity).collides(reference) or entity['id'] * self.value < reference.id * self.value for reference in reference_entities)]

        elif self.reltype == 'proximity-max':
            entity_centers = [Point.from_model(entity['center']) for entity in entities]
            agreeing_ids = set()
            for reference in reference_entities:
                reference_center = Point.from_model(reference['center'])
                max_distance = -1.0
                max_entity = None
                for entity, entity_center in zip(entities, entity_centers):
                    if entity['id'] == reference['id']:
                        continue
                    distance = (entity_center - reference_center).length() * self.value
                    if distance > max_distance:
                        max_distance = distance
                        max_entity = entity
                if max_entity is not None:
                    agreeing_ids.add(max_entity['id'])
            return [entity for entity in entities if entity['id'] not in agreeing_ids]

        elif self.reltype == 'proximity-rel':
            # min distance to second
            comparison_entities = self.comparison.agreeing_entities(entities=entities)
            entity_centers = [Point.from_model(entity['center']) for entity in entities]
            reference_centers = [Point.from_model(reference['center']) for reference in reference_entities]
            comparison_centers = [Point.from_model(comparison['center']) for comparison in comparison_entities]
            agreeing_ids = set()
            for reference, reference_center, comparison, comparison_center in zip(reference_entities, reference_centers, comparison_entities, comparison_centers):
                if reference['id'] == comparison['id']:
                    continue
                reference_distance = (comparison_center - reference_center).length() * self.value
                for entity, entity_center in zip(entities, entity_centers):
                    if entity['id'] == reference['id'] or entity['id'] == comparison['id']:
                        continue
                    if (entity_center - reference_center).length() * self.value > reference_distance:
                        agreeing_ids.append(entity['id'])
            return [entity for entity in entities if entity['id'] not in agreeing_ids]

        elif self.reltype == 'size-rel':
            reference_areas = [Shape.from_model(reference['shape']).area() for reference in reference_entities]
            return [entity for entity in entities if all(Shape.from_model(entity['shape']).area() * self.value <= reference_area * self.value for reference_area in reference_areas)]

        elif self.reltype == 'shade-rel':
            return [entity for entity in entities if all(entity['color']['shade'] * self.value <= reference['color']['shade'] * self.value for reference in reference_entities)]
