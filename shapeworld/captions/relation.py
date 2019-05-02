from shapeworld.captions import Predicate, EntityType, Selector, Settings


class Relation(Predicate):

    predtypes = {'attribute', 'type', 'negation', 'unspecific', 'x-rel', 'y-rel', 'z-rel', 'proximity-rel', 'size-rel', 'shade-rel', 'shape-rel', 'color-rel', 'texture-rel'}
    meta_relations = {'negation'}
    no_inverse_relations = {'unspecific'}
    ternary_relations = {'proximity-rel'}
    # reference_selectors = {'proximity-two', 'proximity-max'}

    __slots__ = ('predtype', 'value', 'reference', 'comparison')

    def __init__(self, predtype, value, reference=None, comparison=None):
        assert predtype in Relation.predtypes
        if predtype == 'attribute':
            from shapeworld.captions import Attribute
            assert isinstance(value, Attribute)
            assert reference is None and comparison is None
        elif predtype == 'type':
            assert isinstance(value, EntityType)
            assert reference is None and comparison is None
        elif predtype == 'unspecific':
            assert value == 1
            assert isinstance(reference, EntityType)
            assert comparison is None or isinstance(comparison, Selector)
        elif predtype in Relation.meta_relations:
            assert value == 1
            assert isinstance(reference, Relation)
            assert comparison is None
        elif predtype in Relation.ternary_relations:
            assert value == -1 or value == 1
            assert isinstance(reference, EntityType)
            assert isinstance(comparison, Selector)
        else:
            assert value == -1 or value == 1
            assert isinstance(reference, EntityType)
            assert comparison is None or isinstance(comparison, Selector)
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
        elif self.predtype in Relation.ternary_relations:
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                reference=self.reference.model(),
                comparison=self.comparison.model()
            )
        else:
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                reference=self.reference.model()
            )

    def polish_notation(self, reverse=False):
        if reverse:
            if self.predtype in ('attribute', 'type'):
                return self.value.polish_notation(reverse=reverse) + \
                    ['{}-{}'.format(self, self.predtype)]
            elif self.predtype in Relation.ternary_relations:
                return self.reference.polish_notation(reverse=reverse) + \
                    self.comparison.polish_notation(reverse=reverse) + \
                    ['{}-{}-{}'.format(self, self.predtype, self.value)]
            else:
                return self.reference.polish_notation(reverse=reverse) + \
                    ['{}-{}-{}'.format(self, self.predtype, self.value)]
        else:
            if self.predtype in ('attribute', 'type'):
                return ['{}-{}'.format(self, self.predtype)] + \
                    self.value.polish_notation(reverse=reverse)
            elif self.predtype in Relation.ternary_relations:
                return ['{}-{}-{}'.format(self, self.predtype, self.value)] + \
                    self.reference.polish_notation(reverse=reverse) + \
                    self.comparison.polish_notation(reverse=reverse)
            else:
                return ['{}-{}-{}'.format(self, self.predtype, self.value)] + \
                    self.reference.polish_notation(reverse=reverse)

    def apply_to_predication(self, predication):
        if self.predtype in ('attribute', 'type'):
            self.value.apply_to_predication(predication=predication)
            return
        elif self.predtype in Relation.ternary_relations:
            ref_predication = predication.sub_predication(reset=True)
            self.reference.apply_to_predication(predication=ref_predication)
            comp_predication = predication.sub_predication(reset=True)
            self.comparison.apply_to_predication(predication=comp_predication)
        else:
            ref_predication = predication.sub_predication(reset=True)
            self.reference.apply_to_predication(predication=ref_predication)
            comp_predication = None
        predication.apply(predicate=self, ref_predication=ref_predication, comp_predication=comp_predication)
        return ref_predication, comp_predication

    def pred_agreement(self, entity, ref_predication=None, comp_predication=None):
        if self.predtype in ('attribute', 'type'):
            return self.value.pred_agreement(entity=entity)

        elif self.predtype in Relation.meta_relations:
            sub_ref_predication = ref_predication.get_sub_predication(0)
            sub_comp_predication = ref_predication.get_sub_predication(1)

            if self.predtype == 'negation':
                if self.value == -1:
                    return self.reference.pred_disagreement(entity=entity, ref_predication=sub_ref_predication, comp_predication=sub_comp_predication)
                else:
                    return self.reference.pred_agreement(entity=entity, ref_predication=sub_ref_predication, comp_predication=sub_comp_predication)

        ref_entities = ref_predication.agreeing

        if self.predtype == 'unspecific':
            return any(True for reference in ref_entities if reference != entity)

        elif self.predtype == 'x-rel':
            # min distance in case of overlap
            return any((entity.center.x - reference.center.x) * self.value > max(Settings.min_axis_distance, abs(entity.center.y - reference.center.y)) for reference in ref_entities)

        elif self.predtype == 'y-rel':
            return any((entity.center.y - reference.center.y) * self.value > max(Settings.min_axis_distance, abs(entity.center.x - reference.center.x)) for reference in ref_entities)

        elif self.predtype == 'z-rel':
            return any(entity.collides(reference, ratio=True, symmetric=True) > Settings.min_overlap and (entity.id - reference.id) * self.value > 0 for reference in ref_entities)

        elif self.predtype == 'size-rel':
            return any((entity.shape.area - reference.shape.area) * self.value > Settings.min_area for reference in ref_entities if reference.shape == entity.shape)

        elif self.predtype == 'shade-rel':
            return any((entity.color.shade - reference.color.shade) * self.value > Settings.min_shade for reference in ref_entities if reference.color == entity.color)

        elif self.predtype == 'shape-rel':
            return any(entity.shape == reference.shape if self.value == 1 else entity.shape != reference.shape for reference in ref_entities if reference != entity)

        elif self.predtype == 'color-rel':
            return any(entity.color == reference.color if self.value == 1 else entity.color != reference.color for reference in ref_entities if reference != entity)

        elif self.predtype == 'texture-rel':
            return any(entity.texture == reference.texture if self.value == 1 else entity.texture != reference.texture for reference in ref_entities if reference != entity)

        comp_entities = comp_predication.agreeing

        if self.predtype == 'proximity-rel':
            for reference in ref_entities:
                for comparison in comp_entities:
                    if reference == comparison or entity == reference or entity == comparison:
                        continue
                    if ((entity.center - comparison.center).length() - (reference.center - comparison.center).length()) * self.value > Settings.min_distance:
                        return True
            return False

        else:
            assert False

    def pred_disagreement(self, entity, ref_predication=None, comp_predication=None):
        if self.predtype in ('attribute', 'type'):
            return self.value.pred_disagreement(entity=entity)

        elif self.predtype in Relation.meta_relations:
            sub_ref_predication = ref_predication.get_sub_predication(0)
            sub_comp_predication = ref_predication.get_sub_predication(1)

            if self.predtype == 'negation':
                if self.value == -1:
                    return self.reference.pred_agreement(entity=entity, ref_predication=sub_ref_predication, comp_predication=sub_comp_predication)
                else:
                    return self.reference.pred_disagreement(entity=entity, ref_predication=sub_ref_predication, comp_predication=sub_comp_predication)

        ref_entities = ref_predication.not_disagreeing
        if len(ref_entities) == 1 and ref_entities[0] == entity:
            return False

        if self.predtype == 'unspecific':
            return all(False for reference in ref_entities if reference != entity)

        elif self.predtype == 'x-rel':
            return all((reference.center.x - entity.center.x) * self.value > Settings.min_axis_distance for reference in ref_entities)

        elif self.predtype == 'y-rel':
            return all((reference.center.y - entity.center.y) * self.value > Settings.min_axis_distance for reference in ref_entities)

        elif self.predtype == 'z-rel':
            for reference in ref_entities:
                collision = entity.collides(reference, ratio=True, symmetric=True)
                if collision > 0.0 and (collision <= Settings.min_overlap or entity.id * self.value > reference.id * self.value):
                    return False
            return True

        elif self.predtype == 'size-rel':
            return all((reference.shape.area - entity.shape.area) * self.value > Settings.min_area for reference in ref_entities)  # and reference.shape == entity.shape)

        elif self.predtype == 'shade-rel':
            return all((reference.color.shade - entity.color.shade) * self.value > Settings.min_shade for reference in ref_entities)  # and reference.color == entity.color)

        elif self.predtype == 'shape-rel':
            return all(entity.shape != reference.shape if self.value == 1 else entity.shape == reference.shape for reference in ref_entities if reference != entity)

        elif self.predtype == 'color-rel':
            return all(entity.color != reference.color if self.value == 1 else entity.color == reference.color for reference in ref_entities if reference != entity)

        elif self.predtype == 'texture-rel':
            return all(entity.texture != reference.texture if self.value == 1 else entity.texture == reference.texture for reference in ref_entities if reference != entity)

        comp_entities = comp_predication.not_disagreeing
        if len(comp_entities) == 1 and comp_entities[0] == entity:
            return False

        if self.predtype == 'proximity-rel':
            for reference in ref_entities:
                for comparison in comp_entities:
                    if comparison == reference:
                        continue
                    if ((reference.center - comparison.center).length() - (entity.center - comparison.center).length()) * self.value < Settings.min_distance:
                        return False
            return True

        else:
            assert False
