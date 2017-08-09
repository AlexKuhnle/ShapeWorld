from random import choice
from shapeworld import util
from shapeworld.caption import Quantifier
from shapeworld.captioners import WorldCaptioner, AttributesNounCaptioner, AttributesRelationCaptioner


class RelativeQuantifierCaptioner(WorldCaptioner):

    # modes
    # 0: correct
    # 1: incorrect quantifier
    # 2: incorrect restrictor
    # 3: incorrect body

    name = 'relative_quantifier'
    statistics_header = 'qrange,quantity,mode'

    def __init__(self, restrictor_captioner=None, body_captioner=None, incorrect_distribution=None):
        super(RelativeQuantifierCaptioner, self).__init__()
        self.restrictor_captioner = restrictor_captioner or AttributesNounCaptioner(hypernym_ratio=1.0)
        self.body_captioner = body_captioner or AttributesRelationCaptioner()
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution or [2, 1, 1])

    def set_realizer(self, realizer):
        if not super(RelativeQuantifierCaptioner, self).set_realizer(realizer):
            return False
        assert 'relative' in realizer.quantifiers
        self.restrictor_captioner.set_realizer(realizer=realizer)
        self.body_captioner.set_realizer(realizer=realizer)
        self.quantifiers = list((qrange, quantity) for qrange, quantities in realizer.quantifiers['relative'].items() for quantity in quantities)
        assert self.quantifiers
        return True

    def caption_world(self, entities, correct):
        qrange, quantity = choice(self.quantifiers)

        if correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.incorrect_distribution)

        for _ in range(RelativeQuantifierCaptioner.MAX_ATTEMPTS):

            if mode == 2:  # incorrect restrictor
                restrictor = self.restrictor_captioner(entities=entities, correct=False)
            else:
                restrictor = self.restrictor_captioner(entities=entities, correct=True)
            if restrictor is None:
                continue

            if mode == 3:  # incorrect body
                body = self.body_captioner(entities=entities, correct=False)
            else:
                body = self.body_captioner(entities=entities, correct=True)
            if body is None:
                continue

            quantifier = Quantifier(qtype='relative', qrange=qrange, quantity=quantity, restrictor=restrictor, body=body)

            if quantifier.agreement(entities=entities) == float(correct):
                self.report(qrange, quantity, mode)
                return quantifier

        return None
