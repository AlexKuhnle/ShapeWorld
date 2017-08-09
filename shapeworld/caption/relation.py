from shapeworld.caption import Predicate, Noun, Settings


class Relation(Predicate):

    # can be unary, binary, binary with generalized quantification
    # dmrs might need to insert quantifier

    ternary_relations = ('proximity-rel',)

    __slots__ = ('reltype', 'value', 'reference', 'comparison')

    def __init__(self, reltype, value, reference=None, comparison=None):
        assert reltype in ('modifier', 'noun', 'x-rel', 'y-rel', 'z-rel', 'proximity-max', 'proximity-rel', 'size-rel', 'shade-rel')
        assert (reference is None) == (reltype in ('modifier', 'noun')) != isinstance(reference, Noun)
        assert (comparison is None) != (reltype in Relation.ternary_relations) == isinstance(comparison, Noun)
        if reltype == 'modifier':
            from shapeworld.caption import Modifier
            assert isinstance(value, Modifier)
        elif reltype == 'noun':
            assert isinstance(value, Noun)
        elif reltype in ('x-rel', 'y-rel', 'z-rel', 'proximity-max', 'proximity-rel', 'size-rel', 'shade-rel'):
            assert value == -1 or value == 1
        self.reltype = reltype
        self.value = value
        self.reference = reference
        self.comparison = comparison

    def model(self):
        if self.reltype == 'modifier' or self.reltype == 'noun':
            return dict(component='relation', reltype=self.reltype, value=self.value.model())
        elif self.comparison is None:
            return dict(component='relation', reltype=self.reltype, value=self.value, reference=self.reference.model())
        else:
            return dict(component='relation', reltype=self.reltype, value=self.value, reference=self.reference.model(), comparison=self.comparison.model())

    def agreeing_entities(self, entities):
        if self.reltype == 'modifier' or self.reltype == 'noun':
            return self.value.agreeing_entities(entities=entities)

        reference_entities = self.reference.agreeing_entities(entities=entities)

        if self.reltype == 'x-rel':
            # min distance in case of overlap
            return [entity for entity in entities if any((entity.center.x - reference.center.x) * self.value > max(Settings.min_distance, abs(entity.center.y - reference.center.y)) for reference in reference_entities)]

        elif self.reltype == 'y-rel':
            return [entity for entity in entities if any((entity.center.y - reference.center.y) * self.value > max(Settings.min_distance, abs(entity.center.x - reference.center.x)) for reference in reference_entities)]

        elif self.reltype == 'z-rel':
            return [entity for entity in entities if any(entity.collides(reference, ratio=True, symmetric=True) > Settings.min_overlap and entity.id * self.value > reference.id * self.value for reference in reference_entities)]

        elif self.reltype == 'proximity-max':
            # min distance to second
            agreeing_ids = set()
            for reference in reference_entities:
                max_distance = -1.0
                max_entity = None
                significant = False
                for entity in entities:
                    if entity.id == reference.id:
                        continue
                    distance = (entity.center - reference.center).length * self.value
                    if distance > max_distance:
                        significant = (distance - max_distance > Settings.min_distance)
                        max_distance = distance
                        max_entity = entity
                if significant:
                    agreeing_ids.add(max_entity.id)
            return [entity for entity in entities if entity.id in agreeing_ids]

        elif self.reltype == 'proximity-rel':
            # min distance to second
            comparison_entities = self.comparison.agreeing_entities(entities=entities)
            agreeing_ids = set()
            for reference in reference_entities:
                for comparison in comparison_entities:
                    if reference.id == comparison.id:
                        continue
                    reference_distance = (comparison.center - reference.center).length * self.value
                    for entity in entities:
                        if entity.id == reference.id or entity.id == comparison.id:
                            continue
                        if (entity.center - reference.center).length * self.value - reference_distance > Settings.min_distance:
                            agreeing_ids.add(entity.id)
            return [entity for entity in entities if entity.id in agreeing_ids]

        elif self.reltype == 'size-rel':
            # min difference
            return [entity for entity in entities if any(entity.shape.area * self.value - reference.shape.area * self.value > Settings.min_area for reference in reference_entities)]

        elif self.reltype == 'shade-rel':
            # min difference
            return [entity for entity in entities if any(entity.color.shade * self.value - reference.color.shade * self.value > Settings.min_shade for reference in reference_entities)]

    def disagreeing_entities(self, entities):
        if self.reltype == 'modifier' or self.reltype == 'noun':
            return self.value.disagreeing_entities(entities=entities)

        reference_entities = self.reference.agreeing_entities(entities=entities)

        if self.reltype == 'x-rel':
            return [entity for entity in entities if all((entity.center.x - reference.center.x) * self.value < 0 for reference in reference_entities)]

        elif self.reltype == 'y-rel':
            return [entity for entity in entities if all((entity.center.y - reference.center.y) * self.value < 0 for reference in reference_entities)]

        elif self.reltype == 'z-rel':
            return [entity for entity in entities if any(entity.collides(reference) for reference in reference_entities) and all(not entity.collides(reference) or entity.id * self.value < reference.id * self.value for reference in reference_entities)]

        elif self.reltype == 'proximity-max':
            not_disagreeing_ids = set()
            for reference in reference_entities:
                max_distance = -1.0
                max_entities = list()
                for entity in entities:
                    if entity.id == reference.id:
                        continue
                    distance = (entity.center - reference.center).length * self.value
                    if distance > max_distance:
                        max_distance = distance
                        max_entities = [(e, d) for e, d in max_entities if max_distance - d < Settings.min_distance]
                        max_entities.append((entity, max_distance))
                not_disagreeing_ids.update(e.id for e, d in max_entities)
            return [entity for entity in entities if entity.id not in not_disagreeing_ids]

        elif self.reltype == 'proximity-rel':
            # min distance to second
            comparison_entities = self.comparison.agreeing_entities(entities=entities)
            not_disagreeing_ids = set()
            for reference in reference_entities:
                for comparison in comparison_entities:
                    if reference.id == comparison.id:
                        continue
                    reference_distance = (comparison.center - reference.center).length * self.value
                    for entity in entities:
                        if entity.id == reference.id or entity.id == comparison.id:
                            continue
                        if reference_distance - (entity.center - reference.center).length * self.value > Settings.min_distance:
                            not_disagreeing_ids.add(entity.id)
            return [entity for entity in entities if entity.id not in not_disagreeing_ids]

        elif self.reltype == 'size-rel':
            return [entity for entity in entities if all(reference.shape.area * self.value - entity.shape.area * self.value > Settings.min_area for reference in reference_entities)]

        elif self.reltype == 'shade-rel':
            return [entity for entity in entities if all(reference.color.shade * self.value - entity.color.shade * self.value > Settings.min_area for reference in reference_entities)]
