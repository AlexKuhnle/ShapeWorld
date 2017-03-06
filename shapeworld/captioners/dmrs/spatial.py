from copy import deepcopy
from random import choice, random
from shapeworld.captioners.dmrs_captioner import AnchorDmrs, DmrsCaption, DmrsCaptioner, DmrsCaptionerComponent


class SpatialDmrsCaptioner(DmrsCaptioner):

    def __init__(self, caption_size, words):
        super().__init__(caption_size, words)
        self.dmrs = {
            'left': AnchorDmrs.parse('[noun]:pred x? <-1- ***[verb]:_to_p e[ppi--] -2-> _left_n_of x[_s___] <-- _the_q; :_left_n_of -1-> [arg]:pred x[?s???] <-- _a_q')[0],
            'right': AnchorDmrs.parse('[noun]:pred x? <-1- ***[verb]:_to_p e[ppi--] -2-> _right_n_of x[_s___] <-- _the_q; :_right_n_of -1-> [arg]:pred x[?s???] <-- _a_q')[0],
            'up': AnchorDmrs.parse('[noun]:pred x? <-1- ***[verb]:_above_p e[ppi--] -2-> [arg]:pred x[?s???] <-- _a_q')[0],
            'down': AnchorDmrs.parse('[noun]:pred x? <-1- ***[verb]:_below_p e[ppi--] -2-> [arg]:pred x[?s???] <-- _a_q')[0]}

    def instantiate(self, arg, direction):
        assert direction in self.dmrs

        def agreeing_entities(entities):
            references = arg.agreeing_entities(entities)
            if direction == 'left':
                entities = [entity for entity in entities if any(reference.center.x - entity.center.x > abs(entity.center.y - reference.center.y) for reference in references)]
            elif direction == 'right':
                entities = [entity for entity in entities if any(entity.center.x - reference.center.x > abs(entity.center.y - reference.center.y) for reference in references)]
            elif direction == 'up':
                entities = [entity for entity in entities if any(reference.center.y - entity.center.y > abs(entity.center.x - reference.center.x) for reference in references)]
            else:
                entities = [entity for entity in entities if any(entity.center.y - reference.center.y > abs(entity.center.x - reference.center.x) for reference in references)]
            return entities

        dmrs = deepcopy(self.dmrs[direction]).compose(arg.dmrs, {'arg': 'noun'})
        return DmrsCaption('relation', agreeing_entities, DmrsCaption.none_agreement, dmrs)

    def caption_world(self, world):
        reference = choice(world.entities)
        arg = DmrsCaptionerComponent.nouns[str(reference.shape)].instantiate((DmrsCaptionerComponent.modifiers[str(reference.color)].instantiate(),))
        directions = ['left', 'right', 'up', 'down']
        direction = choice(directions)
        directions.remove(direction)
        relation = self.instantiate(arg, direction)
        while not relation.agreeing_entities(world.entities):
            direction = choice(directions)
            directions.remove(direction)
            relation = self.instantiate(arg, direction)
        target = choice(relation.agreeing_entities(world.entities))
        rstr = DmrsCaptionerComponent.nouns[str(target.shape)].instantiate((DmrsCaptionerComponent.modifiers[str(target.color)].instantiate(),))
        quantifier = DmrsCaptionerComponent.quantifiers['a'].instantiate(rstr, relation)
        return quantifier
