from shapeworld import util
from shapeworld.captions import Settings, Predicate, EntityType


class Selector(Predicate):

    predtypes = {'unique', 'x-two', 'y-two', 'size-two', 'shade-two', 'x-max', 'y-max', 'proximity-two', 'proximity-max', 'size-max', 'shade-max'}
    reference_selectors = {'proximity-two', 'proximity-max'}

    def __init__(self, predtype, value=None, scope=None, reference=None):
        assert predtype in Selector.predtypes
        if predtype == 'unique':
            assert value is None
        else:
            assert value == -1 or value == 1
        assert isinstance(scope, EntityType)
        if predtype in Selector.reference_selectors:
            assert isinstance(reference, Selector)
        else:
            assert reference is None or isinstance(reference, Selector)
        super(Selector, self).__init__(predtype=predtype, value=value)
        self.scope = scope
        self.reference = reference

    def model(self):
        if self.predtype == 'unique':
            return dict(
                component=str(self),
                predtype=self.predtype,
                scope=self.scope.model()
            )
        elif self.predtype in Selector.reference_selectors:
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                scope=self.scope.model(),
                reference=self.reference.model()
            )
        else:
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                scope=self.scope.model()
            )

    def reverse_polish_notation(self):
        return ['{}-{}-{}'.format(self, self.predtype, self.value)]

    def apply_to_predication(self, predication):
        if self.predtype in Selector.reference_selectors:
            self.scope.apply_to_predication(predication=predication)
            scope_predication = predication.copy()
            ref_predication = predication.sub_predication(reset=True)
            self.reference.apply_to_predication(predication=ref_predication)
        else:
            self.scope.apply_to_predication(predication=predication)
            scope_predication = predication.copy()
            ref_predication = None
        predication.apply(predicate=self, scope_predication=scope_predication, ref_predication=ref_predication)

    def pred_agreement(self, entity, scope_predication, ref_predication=None):
        scope_entities = scope_predication.not_disagreeing

        if all(other != entity for other in scope_entities):
            return False

        elif self.predtype == 'unique':
            return len(scope_entities) == 1 and len(scope_predication.agreeing) == 1

        elif len(scope_predication.agreeing) < 2:
            return False

        elif self.predtype[-4:] == '-two' and (len(scope_entities) != 2 or len(scope_predication.agreeing) != 2):
            # print('c', len(scope_entities), len(scope_predication.agreeing))
            return False

        elif self.predtype == 'x-two' or self.predtype == 'x-max':
            return util.all_and_any((entity.center.x - other.center.x) * self.value > Settings.min_axis_distance for other in scope_entities if other != entity)

        elif self.predtype == 'y-two' or self.predtype == 'y-max':
            return util.all_and_any((entity.center.y - other.center.y) * self.value > Settings.min_axis_distance for other in scope_entities if other != entity)

        elif self.predtype == 'size-two' or self.predtype == 'size-max':
            return util.all_and_any((entity.shape.area - other.shape.area) * self.value > Settings.min_area for other in scope_entities if other != entity and other.shape == entity.shape)

        elif self.predtype == 'shade-two' or self.predtype == 'shade-max':
            return util.all_and_any((entity.color.shade - other.color.shade) * self.value > Settings.min_shade for other in scope_entities if other != entity and other.color == entity.color)

        ref_entities = ref_predication.agreeing

        if self.predtype == 'proximity-two' or self.predtype == 'proximity-max':
            for other in scope_entities:
                for reference in ref_entities:
                    if other == reference or entity == other or entity == reference:
                        continue
                    if ((entity.center - other.center).length() - (reference.center - other.center).length()) * self.value > Settings.min_distance:
                        return True
            return False

        else:
            assert False

    def pred_disagreement(self, entity, scope_predication, ref_predication=None):
        scope_entities = scope_predication.agreeing

        if self.predtype == 'unique':
            return False

        elif len(scope_entities) < 2:
            return False

        elif self.predtype[-4:] == '-two' and (len(scope_entities) != 2 or len(scope_predication.not_disagreeing) != 2):
            # print('i', len(scope_entities), len(scope_predication.agreeing))
            return False

        elif self.predtype == 'x-two' or self.predtype == 'x-max':
            return any((other.center.x - entity.center.x) * self.value > Settings.min_axis_distance for other in scope_entities)

        elif self.predtype == 'y-two' or self.predtype == 'y-max':
            return any((other.center.y - entity.center.y) * self.value > Settings.min_axis_distance for other in scope_entities)

        elif self.predtype == 'size-two' or self.predtype == 'size-max':
            return any((other.shape.area - entity.shape.area) * self.value > Settings.min_area for other in scope_entities)  # if other.shape == entity.shape)

        elif self.predtype == 'shade-two' or self.predtype == 'shade-max':
            return any((other.color.shade - entity.color.shade) * self.value > Settings.min_shade for other in scope_entities)  # if other.color == entity.color)

        ref_entities = ref_predication.not_disagreeing
        if len(ref_entities) == 1 and ref_entities[0] == entity:
            return False

        if self.predtype == 'proximity-two' or self.predtype == 'proximity-max':
            for other in scope_entities:
                for reference in ref_entities:
                    if reference == other:
                        continue
                    if ((reference.center - other.center).length() - (entity.center - other.center).length()) * self.value < Settings.min_distance:
                        return False
            return True

        else:
            assert False
