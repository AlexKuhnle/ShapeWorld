from copy import deepcopy
from random import choice
from shapeworld.captioners.dmrs_captioner import AnchorDmrs, DmrsCaption, DmrsCaptioner, DmrsCaptionerComponent


class ExistentialDmrsCaptioner(DmrsCaptioner):

    def __init__(self, caption_size, words):
        super().__init__(caption_size, words)
        self.dmrs = AnchorDmrs.parse('[noun]:node <-1- ***[verb]:_be_v_there e[ppi--]')[0]

    def instantiate(self):
        category = 'relation'
        dmrs = deepcopy(self.dmrs)
        return DmrsCaption(category, DmrsCaption.all_agreeing_entities, DmrsCaption.none_agreement, dmrs)

    def caption_world(self, world):
        entity = choice(world.entities)
        shape = str(entity.shape)
        color = str(entity.fill.color)
        rstr = DmrsCaptionerComponent.nouns[shape].instantiate((DmrsCaptionerComponent.modifiers[color].instantiate(),))
        body = self.instantiate()
        quantifier = DmrsCaptionerComponent.quantifiers['a'].instantiate(rstr, body)
        return quantifier
