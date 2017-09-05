from random import choice
from shapeworld import util
from shapeworld.caption import Quantifier
from shapeworld.captioners import WorldCaptioner, AttributesTypeCaptioner, AttributesRelationCaptioner


class RelativeQuantifierCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect quantifier
    # 2: incorrect restrictor
    # 3: incorrect body

    def __init__(self, restrictor_captioner=None, body_captioner=None, incorrect_distribution=None, trivial_acceptance_rate=None):
        self.restrictor_captioner = util.value_or_default(restrictor_captioner, AttributesTypeCaptioner())
        self.body_captioner = util.value_or_default(body_captioner, AttributesRelationCaptioner())
        super(RelativeQuantifierCaptioner, self).__init__(internal_captioners=(self.restrictor_captioner, self.body_captioner), trivial_acceptance_rate=trivial_acceptance_rate)
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [2, 1, 1]))

    def set_realizer(self, realizer):
        if not super(RelativeQuantifierCaptioner, self).set_realizer(realizer):
            return False
        assert 'relative' in realizer.quantifiers
        self.restrictor_captioner.set_realizer(realizer=realizer)
        self.body_captioner.set_realizer(realizer=realizer)
        self.quantifiers = list((qrange, quantity) for qrange, quantities in realizer.quantifiers['relative'].items() for quantity in quantities)
        assert self.quantifiers
        return True

    def sample_values(self, mode, correct):
        super(RelativeQuantifierCaptioner, self).sample_values(mode=mode, correct=correct)

        self.qrange, self.quantity = choice(self.quantifiers)
        if self.quantity == 0.0 and (self.qrange == 'eq' or self.qrange == 'leq'):
            self.incorrect_mode = 0 if not correct else 1 + util.sample(self.incorrect_distribution)
        else:
            self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        self.restrictor_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 2))  # 2: incorrect restrictor
        self.body_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 3))  # 3: incorrect body

        if self.incorrect_mode == 1:  # 1: incorrect quantifier
            self.incorrect_quantifiers = [(qrange, quantity) for qrange, quantity in self.quantifiers if qrange != self.qrange or quantity != self.quantity]

    def model(self):
        return util.merge_dicts(
            dict1=super(RelativeQuantifierCaptioner, self).model(),
            dict2=dict(qrange=self.qrange, quantity=self.quantity, incorrect_mode=self.incorrect_mode, restrictor_captioner=self.restrictor_captioner.model(), body_captioner=self.body_captioner.model())
        )

    def caption_world(self, entities, relevant_entities):
        body = self.body_captioner(entities=entities, relevant_entities=relevant_entities)
        if body is None:
            return None

        if self.incorrect_mode == 3:  # 3: incorrect body
            relevant_entities_restrictor = body.disagreeing_entities(entities=relevant_entities)
        else:
            relevant_entities_restrictor = body.agreeing_entities(entities=relevant_entities)

        restrictor = self.restrictor_captioner(entities=entities, relevant_entities=relevant_entities_restrictor)
        if restrictor is None:
            return None

        quantifier = Quantifier(qtype='relative', qrange=self.qrange, quantity=self.quantity, restrictor=restrictor, body=body)

        if self.incorrect_mode == 1:  # 1: incorrect quantifier
            if (quantifier.agreement(entities=relevant_entities) > 0.0 and self.correct) or (quantifier.agreement(entities=relevant_entities) < 0.0 and not self.correct):
                quantifier.qrange, quantifier.quantity = choice(self.incorrect_quantifiers)
                return quantifier
            else:
                return None

        else:
            return quantifier
