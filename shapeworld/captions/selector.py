from shapeworld import util
from shapeworld.captions import Settings, Predicate, EntityType


class Selector(Predicate):

    predtypes = {'unique', 'x-two', 'y-two', 'proximity-two', 'size-two', 'shade-two', 'x-max', 'y-max', 'proximity-max', 'size-max', 'shade-max'}
    comparison_selectors = {'proximity-two', 'proximity-max'}

    def __init__(self, predtype, value=None, scope=None, comparison=None):
        assert predtype in Selector.predtypes
        if predtype == 'unique':
            assert value is None
        else:
            assert value == -1 or value == 1
        assert isinstance(scope, EntityType)
        if predtype in Selector.comparison_selectors:
            assert isinstance(comparison, Selector)
        else:
            assert comparison is None or isinstance(comparison, Selector)
        super(Selector, self).__init__(predtype=predtype, value=value)
        self.scope = scope
        self.comparison = comparison

    def model(self):
        if self.predtype == 'unique':
            return dict(
                component=str(self),
                predtype=self.predtype,
                scope=self.scope.model()
            )
        elif self.predtype in Selector.comparison_selectors:
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                scope=self.scope.model(),
                comparison=self.comparison.model()
            )
        else:
            return dict(
                component=str(self),
                predtype=self.predtype,
                value=self.value,
                scope=self.scope.model()
            )

    def polish_notation(self, reverse=False):
        if reverse:
            if self.predtype == 'unique':
                return self.scope.polish_notation(reverse=reverse) + \
                    ['{}-{}'.format(self, self.predtype)]
            elif self.predtype in Selector.comparison_selectors:
                return self.scope.polish_notation(reverse=reverse) + \
                    self.comparison.polish_notation(reverse=reverse) + \
                    ['{}-{}-{}'.format(self, self.predtype, self.value)]
            else:
                return self.scope.polish_notation(reverse=reverse) + \
                    ['{}-{}-{}'.format(self, self.predtype, self.value)]
        else:
            if self.predtype == 'unique':
                return ['{}-{}'.format(self, self.predtype)] + \
                    self.scope.polish_notation(reverse=reverse)
            elif self.predtype in Selector.comparison_selectors:
                return ['{}-{}-{}'.format(self, self.predtype, self.value)] + \
                    self.scope.polish_notation(reverse=reverse) + \
                    self.comparison.polish_notation(reverse=reverse)
            else:
                return ['{}-{}-{}'.format(self, self.predtype, self.value)] + \
                    self.scope.polish_notation(reverse=reverse)

    def apply_to_predication(self, predication):
        self.scope.apply_to_predication(predication=predication)
        scope_predication = predication.sub_predication(reset=True)
        self.scope.apply_to_predication(predication=scope_predication)
        if self.predtype in Selector.comparison_selectors:
            comp_predication = predication.sub_predication(reset=True)
            self.comparison.apply_to_predication(predication=comp_predication)
        else:
            comp_predication = None
        predication.apply(predicate=self, scope_predication=scope_predication, comp_predication=comp_predication)

    def pred_agreement(self, entity, scope_predication, comp_predication=None):
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

        elif self.predtype == 'x-two':
            return util.all_and_any((entity.center.x - other.center.x) * self.value > max(Settings.min_axis_distance, abs(entity.center.y - other.center.y)) for other in scope_entities if other != entity)

        elif self.predtype == 'x-max':
            return util.all_and_any((entity.center.x - other.center.x) * self.value > Settings.min_axis_distance for other in scope_entities if other != entity)

        elif self.predtype == 'y-two':
            return util.all_and_any((entity.center.y - other.center.y) * self.value > max(Settings.min_axis_distance, abs(entity.center.x - other.center.x)) for other in scope_entities if other != entity)

        elif self.predtype == 'y-max':
            return util.all_and_any((entity.center.y - other.center.y) * self.value > Settings.min_axis_distance for other in scope_entities if other != entity)

        elif self.predtype == 'size-two' or self.predtype == 'size-max':
            return util.all_and_any((entity.shape.area - other.shape.area) * self.value > Settings.min_area for other in scope_entities if other != entity and other.shape == entity.shape)

        elif self.predtype == 'shade-two' or self.predtype == 'shade-max':
            return util.all_and_any((entity.color.shade - other.color.shade) * self.value > Settings.min_shade for other in scope_entities if other != entity and other.color == entity.color)

        comp_entities = comp_predication.agreeing

        if self.predtype == 'proximity-two' or self.predtype == 'proximity-max':
            for other in scope_entities:
                for comparison in comp_entities:
                    if other == comparison or entity == other or entity == comparison:
                        continue
                    if ((entity.center - comparison.center).length() - (other.center - comparison.center).length()) * self.value > Settings.min_distance:
                        return True
            return False

        else:
            assert False

    def pred_disagreement(self, entity, scope_predication, comp_predication=None):
        scope_entities = scope_predication.agreeing

        if self.predtype == 'unique':
            return False

        elif len(scope_entities) < 2:
            return False

        elif self.predtype[-4:] == '-two' and (len(scope_entities) != 2 or len(scope_predication.not_disagreeing) != 2):
            # print('i', len(scope_entities), len(scope_predication.agreeing))
            return False

        elif all(other != entity for other in scope_predication.not_disagreeing):
            print('!!!!!!')
            return True

        elif self.predtype == 'x-two' or self.predtype == 'x-max':
            return any((other.center.x - entity.center.x) * self.value > Settings.min_axis_distance for other in scope_entities)

        elif self.predtype == 'y-two' or self.predtype == 'y-max':
            return any((other.center.y - entity.center.y) * self.value > Settings.min_axis_distance for other in scope_entities)

        elif self.predtype == 'size-two' or self.predtype == 'size-max':
            return any((other.shape.area - entity.shape.area) * self.value > Settings.min_area for other in scope_entities)  # if other.shape == entity.shape)

        elif self.predtype == 'shade-two' or self.predtype == 'shade-max':
            return any((other.color.shade - entity.color.shade) * self.value > Settings.min_shade for other in scope_entities)  # if other.color == entity.color)

        comp_entities = comp_predication.not_disagreeing
        if len(comp_entities) == 1 and comp_entities[0] == entity:
            return False

        if self.predtype == 'proximity-two' or self.predtype == 'proximity-max':
            for other in scope_entities:
                for comparison in comp_entities:
                    if comparison == other:
                        continue
                    if ((other.center - comparison.center).length() - (entity.center - comparison.center).length()) * self.value < Settings.min_distance:
                        return False
            return True

        else:
            assert False
