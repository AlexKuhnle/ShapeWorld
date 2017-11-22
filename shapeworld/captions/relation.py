from shapeworld import util
from shapeworld.captions import Predicate, EntityType, Settings


class Relation(Predicate):

    predtypes = {'attribute', 'type', 'x-rel', 'y-rel', 'z-rel', 'size-rel', 'shade-rel', 'proximity-rel'}
    ternary_relations = {'proximity-rel'}

    __slots__ = ('predtype', 'value', 'reference', 'comparison')

    def __init__(self, predtype, value, reference=None, comparison=None):
        assert predtype in Relation.predtypes
        assert (reference is None) == (predtype in ('attribute', 'type'))
        assert (reference is not None) == isinstance(reference, EntityType)
        assert comparison is None or isinstance(comparison, EntityType)
        if predtype == 'attribute':
            from shapeworld.captions import Attribute
            assert isinstance(value, Attribute)
        elif predtype == 'type':
            assert isinstance(value, EntityType)
        elif predtype in ('x-rel', 'y-rel', 'z-rel', 'proximity-rel', 'size-rel', 'shade-rel'):
            assert value == -1 or value == 1
        else:
            assert False
        super(Relation, self).__init__(predtype=predtype, value=value)
        self.reference = reference
        self.comparison = comparison

    def model(self):
        if self.predtype in ('attribute', 'type'):
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value.model()
            )
        elif self.predtype in ('x-rel', 'y-rel', 'z-rel', 'size-rel', 'shade-rel'):
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                reference=self.reference.model()
            )
        elif self.predtype == 'proximity-rel':
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                reference=self.reference.model(),
                comparison=self.comparison.model()
            )
        else:
            assert False

    def reverse_polish_notation(self):
        if self.predtype in ('attribute', 'type'):
            return self.value.reverse_polish_notation() + ['{}-{}'.format(self, self.predtype)]
        elif self.predtype in ('x-rel', 'y-rel', 'z-rel', 'size-rel', 'shade-rel'):
            return self.reference.reverse_polish_notation() + ['{}-{}-{}'.format(self, self.predtype, self.value)]
        elif self.predtype == 'proximity-rel':
            return self.reference.reverse_polish_notation() + \
                self.comparison.reverse_polish_notation() + \
                ['{}-{}-{}'.format(self, self.predtype, self.value)]
        else:
            assert False

    def pred_agreement(self, entity, predication):
        if self.predtype in ('attribute', 'type'):
            return self.value.pred_agreement(entity=entity, predication=predication)

        # sub-optimal to do this for every entity again... batched pred_agreement?
        ref_entities = self.reference.filter_agreement(entities=predication.agreeing, predication=predication)

        if self.predtype == 'x-rel':
            # min distance in case of overlap
            return any((entity.center.x - reference.center.x) * self.value > max(Settings.min_distance, abs(entity.center.y - reference.center.y)) for reference in ref_entities if reference != entity)

        elif self.predtype == 'y-rel':
            return any((entity.center.y - reference.center.y) * self.value > max(Settings.min_distance, abs(entity.center.x - reference.center.x)) for reference in ref_entities if reference != entity)

        elif self.predtype == 'z-rel':
            return any(entity.collides(reference, ratio=True, symmetric=True) > Settings.min_overlap and (entity.id - reference.id) * self.value > 0 for reference in ref_entities if reference != entity)

        elif self.predtype == 'size-rel':
            return any((entity.shape.area - reference.shape.area) * self.value > Settings.min_area for reference in ref_entities if reference != entity)

        elif self.predtype == 'shade-rel':
            return any((entity.color.shade - reference.color.shade) * self.value > Settings.min_shade for reference in ref_entities if reference != entity and reference.color == entity.color)

        comp_entities = self.comparison.filter_agreement(entities=predication.agreeing, predication=predication)

        if self.predtype == 'proximity-rel':
            for reference in ref_entities:
                for comparison in comp_entities:
                    if reference == comparison or entity == reference or entity == comparison:
                        continue
                    if ((entity.center - reference.center).length - (comparison.center - reference.center).length) * self.value > Settings.min_distance:
                        return True
            return False

    def pred_disagreement(self, entity, predication):
        if self.predtype in ('attribute', 'type'):
            return self.value.pred_disagreement(entity=entity, predication=predication)

        ref_entities = self.reference.filter_agreement(entities=predication.not_disagreeing, predication=predication)
        if len(ref_entities) == 0:
            return True

        if self.predtype == 'x-rel':
            return util.all_and_any((entity.center.x - reference.center.x) * self.value < 0.0 for reference in ref_entities if reference != entity)

        elif self.predtype == 'y-rel':
            return util.all_and_any((entity.center.y - reference.center.y) * self.value < 0.0 for reference in ref_entities if reference != entity)

        elif self.predtype == 'z-rel':
            for reference in ref_entities:
                collision = entity.collides(reference, ratio=True, symmetric=True)
                if collision > 0.0 and (collision <= Settings.min_overlap or entity.id * self.value > reference.id * self.value):
                    return False
            return True

        elif self.predtype == 'size-rel':
            return util.all_and_any((reference.shape.area - entity.shape.area) * self.value > Settings.min_area for reference in ref_entities if reference != entity)

        elif self.predtype == 'shade-rel':
            return util.all_and_any((reference.color.shade - entity.color.shade) * self.value > Settings.min_shade for reference in ref_entities if reference != entity and reference.color == entity.color)

        comp_entities = self.comparison.filter_agreement(entities=predication.not_disagreeing, predication=predication)

        if self.predtype == 'proximity-rel':
            for reference in ref_entities:
                for comparison in comp_entities:
                    if reference.id == comparison.id or entity.id == reference.id or entity.id == comparison.id:
                        continue
                    if ((comparison.center - reference.center).length - (entity.center - reference.center).length) * self.value < Settings.min_distance:
                        return False
            return True

    # def agreeing_entities(self, entities, world_entities):
    #     if self.predtype == 'attribute' or self.predtype == 'type':
    #         return self.value.agreeing_entities(entities=entities, world_entities=world_entities)

    #     reference_entities = self.reference.agreeing_entities(entities=world_entities, world_entities=world_entities)

    #     if len(reference_entities) == 0:
    #         return list()

    #     if self.predtype == 'x-rel':
    #         # min distance in case of overlap
    #         return [entity for entity in entities if any((entity.center.x - reference.center.x) * self.value > max(Settings.min_distance, abs(entity.center.y - reference.center.y)) for reference in reference_entities if reference != entity)]

    #     elif self.predtype == 'y-rel':
    #         return [entity for entity in entities if any((entity.center.y - reference.center.y) * self.value > max(Settings.min_distance, abs(entity.center.x - reference.center.x)) for reference in reference_entities if reference != entity)]

    #     elif self.predtype == 'z-rel':
    #         return [entity for entity in entities if any(entity.collides(reference, ratio=True, symmetric=True) > 0.0 and (entity.id - reference.id) * self.value > 0 for reference in reference_entities if reference != entity)]

    #     elif self.predtype == 'proximity-max':
    #         # min distance to second
    #         agreeing_ids = set()
    #         for reference in reference_entities:
    #             max_distance = -1.0
    #             max_entity = None
    #             significant = False
    #             for entity in entities:
    #                 if entity.id == reference.id:
    #                     continue
    #                 distance = (entity.center - reference.center).length * self.value
    #                 if distance > max_distance:
    #                     significant = (distance - max_distance) > Settings.min_distance
    #                     max_distance = distance
    #                     max_entity = entity
    #             if significant:
    #                 agreeing_ids.add(max_entity.id)
    #         return [entity for entity in entities if entity.id in agreeing_ids]

    #     elif self.predtype == 'proximity-rel':
    #         # min distance to second
    #         comparison_entities = self.comparison.agreeing_entities(entities=world_entities, world_entities=world_entities)
    #         agreeing_ids = set()
    #         for reference in reference_entities:
    #             for comparison in comparison_entities:
    #                 if reference.id == comparison.id:
    #                     continue
    #                 reference_distance = (comparison.center - reference.center).length * self.value
    #                 for entity in entities:
    #                     if entity.id == reference.id or entity.id == comparison.id:
    #                         continue
    #                     if (entity.center - reference.center).length * self.value - reference_distance > Settings.min_distance:
    #                         agreeing_ids.add(entity.id)
    #         return [entity for entity in entities if entity.id in agreeing_ids]

    #     elif self.predtype == 'size-rel':
    #         # min difference
    #         return [entity for entity in entities if any((entity.shape.area - reference.shape.area) * self.value > Settings.min_area for reference in reference_entities if reference != entity)]  # and reference.shape == entity.shape

    #     elif self.predtype == 'shade-rel':
    #         # min difference
    #         return [entity for entity in entities if any((entity.color.shade - reference.color.shade) * self.value > Settings.min_shade for reference in reference_entities if reference != entity)]  # and reference.color == entity.color

    # def disagreeing_entities(self, entities, world_entities):
    #     if self.predtype == 'attribute' or self.predtype == 'type':
    #         return self.value.disagreeing_entities(entities=entities, world_entities=world_entities)

    #     reference_entities = self.reference.agreeing_entities(entities=world_entities, world_entities=world_entities)

    #     if len(reference_entities) == 0:
    #         return list(entities)

    #     if self.predtype == 'x-rel':
    #         return [entity for entity in entities if util.all_and_any((entity.center.x - reference.center.x) * self.value < 0.0 for reference in reference_entities if reference != entity)]

    #     elif self.predtype == 'y-rel':
    #         return [entity for entity in entities if util.all_and_any((entity.center.y - reference.center.y) * self.value < 0.0 for reference in reference_entities if reference != entity)]

    #     elif self.predtype == 'z-rel':
    #         not_disagreeing_ids = set()
    #         for entity in entities:
    #             for reference in reference_entities:
    #                 collision = entity.collides(reference, ratio=True, symmetric=True)
    #                 if collision > 0.0 and (collision <= Settings.min_overlap or entity.id * self.value > reference.id * self.value):
    #                     not_disagreeing_ids.add(entity.id)
    #                     break
    #         return [entity for entity in entities if entity.id not in not_disagreeing_ids]

    #     elif self.predtype == 'proximity-max':
    #         not_disagreeing_ids = set()
    #         for reference in reference_entities:
    #             max_distance = -1.0
    #             max_entities = list()
    #             for entity in entities:
    #                 if entity.id == reference.id:
    #                     continue
    #                 distance = (entity.center - reference.center).length * self.value
    #                 if distance > max_distance:
    #                     max_distance = distance
    #                     max_entities = [(e, d) for e, d in max_entities if max_distance - d < Settings.min_distance]
    #                 if max_distance - distance < Settings.min_distance:
    #                     max_entities.append((entity, distance))
    #             not_disagreeing_ids.update(e.id for e, _ in max_entities)
    #         return [entity for entity in entities if entity.id not in not_disagreeing_ids]

    #     elif self.predtype == 'proximity-rel':
    #         # min distance to second
    #         comparison_entities = self.comparison.agreeing_entities(entities=world_entities, world_entities=world_entities)
    #         not_disagreeing_ids = set()
    #         for reference in reference_entities:
    #             for comparison in comparison_entities:
    #                 if reference.id == comparison.id:
    #                     continue
    #                 reference_distance = (comparison.center - reference.center).length * self.value
    #                 for entity in entities:
    #                     if entity.id == reference.id or entity.id == comparison.id:
    #                         continue
    #                     if reference_distance - (entity.center - reference.center).length * self.value < Settings.min_distance:
    #                         not_disagreeing_ids.add(entity.id)
    #         return [entity for entity in entities if entity.id not in not_disagreeing_ids]

    #     elif self.predtype == 'size-rel':
    #         return [entity for entity in entities if util.all_and_any((reference.shape.area - entity.shape.area) * self.value > Settings.min_area for reference in reference_entities if reference != entity)]  # and reference.shape == entity.shape

    #     elif self.predtype == 'shade-rel':
    #         return [entity for entity in entities if util.all_and_any((reference.color.shade - entity.color.shade) * self.value > Settings.min_shade for reference in reference_entities if reference != entity)]  # and reference.color == entity.color
