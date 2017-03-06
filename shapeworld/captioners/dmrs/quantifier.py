from copy import deepcopy
from random import choice, random, randrange
from shapeworld.captioners.dmrs_captioner import AnchorDmrs, DmrsCaption, DmrsCaptioner, DmrsCaptionerComponent


class QuantifierDmrsCaptioner(DmrsCaptioner):

    def __init__(self, caption_size, words, shape_names, color_names):
        super().__init__(caption_size, words)
        self.shape_names = shape_names
        self.color_names = color_names
        self.noun_sg_dmrs = AnchorDmrs.parse('[noun]:pred x? <-1- ***[verb]:_be_v_id e[ppi--] -2-> [arg]:pred x[?s???] <-- _a_q')[0]  # ???
        self.noun_pl_dmrs = AnchorDmrs.parse('[noun]:pred x? <-1- ***[verb]:_be_v_id e[ppi--] -2-> [arg]:pred x[?p???] <-- udef_q')[0]
        self.modifier_dmrs = AnchorDmrs.parse('[noun]:node <-1- ***[arg]:pred e[ppi--]')[0]
        self.quantifier_names = ['all', 'every', 'most', 'two', 'some', 'a', 'the', 'no']

    def instantiate(self, arg):  # ???
        assert arg.category in ('noun', 'modifier')
        if arg.category == 'noun':
            dmrs = {True: deepcopy(self.noun_pl_dmrs).compose(arg.dmrs, {'arg': 'noun'}), False: deepcopy(self.noun_sg_dmrs).compose(arg.dmrs, {'arg': 'noun'})}
        else:
            dmrs = deepcopy(self.modifier_dmrs).compose(arg.dmrs, {'arg': 'mod'})
        return DmrsCaption('relation', arg.agreeing_entities, DmrsCaption.none_agreement, dmrs)

    def caption_world(self, world):
        mode = random()
        if mode < 0.5:  # shape is ...
            mode /= 0.5
            subject = DmrsCaptionerComponent.nouns['shape'].instantiate(())
            if mode <= 0.333:  # shape is [shape]
                mode /= 0.333
                if mode <= 0.1:  # potentially non-existent combination for "no"
                    shape = choice(self.shape_names)
                else:
                    shape = choice([str(entity.shape) for entity in world.entities])
                attribute = self.instantiate(DmrsCaptionerComponent.nouns[shape].instantiate(()))
            elif mode <= 0.666:  # shape is [color]
                mode -= 0.333
                mode /= 0.333
                if mode <= 0.1:  # potentially non-existent combination for "no"
                    color = choice(self.color_names)
                else:
                    color = choice([str(entity.color) for entity in world.entities])
                attribute = self.instantiate(DmrsCaptionerComponent.modifiers[color].instantiate())
            else:  # shape is [color] [shape]
                mode -= 0.666
                mode /= 0.334
                if mode <= 0.1:  # potentially non-existent combination for "no"
                    shape = choice(self.shape_names)
                    color = choice(self.color_names)
                else:
                    shape = choice([str(entity.shape) for entity in world.entities])
                    color = choice([str(entity.color) for entity in world.entities])
                attribute = self.instantiate(DmrsCaptionerComponent.nouns[shape].instantiate((DmrsCaptionerComponent.modifiers[color].instantiate(),)))
        else:
            mode -= 0.5
            mode /= 0.5
            if mode < 0.5:  # [shape] is [color]
                mode /= 0.5
                shape = choice([str(entity.shape) for entity in world.entities])
                if mode < 0.1:  # potentially non-existent combination for "no"
                    color = choice(self.color_names)
                else:
                    color = choice([str(entity.color) for entity in world.entities if entity.shape == shape])
                subject = DmrsCaptionerComponent.nouns[shape].instantiate(())
                attribute = self.instantiate(DmrsCaptionerComponent.modifiers[color].instantiate())
            else:  # [color] shape is [shape]
                mode -= 0.5
                mode /= 0.5
                color = choice([str(entity.color) for entity in world.entities])
                if mode < 0.1:  # potentially non-existent combination for "no"
                    shape = choice(self.shape_names)
                else:
                    shape = choice([str(entity.shape) for entity in world.entities if entity.color == color])
                subject = DmrsCaptionerComponent.nouns['shape'].instantiate((DmrsCaptionerComponent.modifiers[color].instantiate(),))
                attribute = self.instantiate(DmrsCaptionerComponent.nouns[shape].instantiate(()))
        before = []
        # bias towards universal quantifiers in first sample
        pick = randrange(len(self.quantifier_names) + 6)
        if pick < 4:
            pick = 0
        elif pick < 8:
            pick = 1
        else:
            pick -= 6
        quantifier_name = self.quantifier_names[pick]
        quantifier = DmrsCaptionerComponent.quantifiers[quantifier_name].instantiate(subject, attribute)
        while not quantifier.agreement(world):
            before.append(pick)
            while True:
                pick = randrange(len(self.quantifier_names))
                if pick not in before:
                    break
            quantifier_name = self.quantifier_names[pick]
            quantifier = DmrsCaptionerComponent.quantifiers[quantifier_name].instantiate(subject, attribute)
        return quantifier
