from shapeworld.caption import Predicate


class Modifier(Predicate):

    __slots__ = ('modtype', 'value')

    def __init__(self, modtype, value=None):
        assert modtype in ('shape', 'color', 'texture', 'combination', 'shapes', 'colors', 'textures', 'combinations')
        self.modtype = modtype
        if modtype in ('shape', 'color', 'texture', 'combination'):
            assert isinstance(value, str)
            self.value = value
        elif modtype in ('shapes', 'colors', 'textures', 'combinations'):
            assert isinstance(value, tuple) or isinstance(value, list)
            self.value = tuple(value)

    def agreeing_entities(self, entities):
        if self.modtype == 'shape':
            return [entity for entity in entities if entity['shape']['name'] == self.value]

        elif self.modtype == 'color':
            return [entity for entity in entities if entity['color']['name'] == self.value]

        elif self.modtype == 'texture':
            return [entity for entity in entities if entity['texture']['name'] == self.value]

        elif self.modtype == 'combination':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) == self.value]

        elif self.modtype == 'shapes':
            return [entity for entity in entities if entity['shape']['name'] in self.value]

        elif self.modtype == 'colors':
            return [entity for entity in entities if entity['color']['name'] in self.value]

        elif self.modtype == 'textures':
            return [entity for entity in entities if entity['texture']['name'] in self.value]

        elif self.modtype == 'combinations':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) in self.value]

        # elif self.modtype == 'shade-max':
        #     # all same color?
        #     max_entity = None
        #     max_shade = -1.0
        #     for entity in entities:
        #         if entity['color']['shade'] * self.value > max_shade:
        #             max_shade = entity['color']['shade'] * self.value
        #             max_entity = entity
        #     if max_entity is None:
        #         return []
        #     else:
        #         return [max_entity]

    def disagreeing_entities(self, entities):
        if self.modtype == 'shape':
            return [entity for entity in entities if entity['shape']['name'] != self.value]

        elif self.modtype == 'color':
            return [entity for entity in entities if entity['color']['name'] != self.value]

        elif self.modtype == 'texture':
            return [entity for entity in entities if entity['texture']['name'] != self.value]

        elif self.modtype == 'combination':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) != self.value]

        elif self.modtype == 'shapes':
            return [entity for entity in entities if entity['shape']['name'] not in self.value]

        elif self.modtype == 'colors':
            return [entity for entity in entities if entity['color']['name'] not in self.value]

        elif self.modtype == 'textures':
            return [entity for entity in entities if entity['texture']['name'] not in self.value]

        elif self.modtype == 'combinations':
            return [entity for entity in entities if (entity['shape']['name'], entity['color']['name'], entity['texture']['name']) not in self.value]
