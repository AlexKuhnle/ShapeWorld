from shapeworld import util
from shapeworld.caption import Existential
from shapeworld.captioners import WorldCaptioner, AttributesNounCaptioner


class ExistentialCaptioner(WorldCaptioner):

    # modes
    # 0: correct
    # 1: incorrect subject
    # 2: incorrect verb

    name = 'generics'
    statistics_header = 'mode'

    def __init__(self, subject_captioner=None, verb_captioner=None, incorrect_distribution=None):
        super(ExistentialCaptioner, self).__init__()
        self.subject_captioner = subject_captioner or AttributesNounCaptioner(hypernym_ratio=1.0)
        self.verb_captioner = verb_captioner or AttributesNounCaptioner()
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution or [1, 1])

    def set_realizer(self, realizer):
        if not super(ExistentialCaptioner, self).set_realizer(realizer=realizer):
            return False
        self.subject_captioner.set_realizer(realizer=realizer)
        self.verb_captioner.set_realizer(realizer=realizer)
        return True

    def caption_world(self, entities, correct):
        if correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.incorrect_distribution)

        for _ in range(ExistentialCaptioner.MAX_ATTEMPTS):

            if mode == 1:  # incorrect subject
                subject = self.subject_captioner(entities=entities, correct=False)
            else:
                subject = self.subject_captioner(entities=entities, correct=True)
            if subject is None:
                continue

            if mode == 2:  # incorrect verb
                verb = self.verb_captioner(entities=entities, correct=False)
            else:
                verb = self.verb_captioner(entities=entities, correct=True)
            if verb is None:
                continue

            generic = Existential(subject=subject, verb=verb)

            if generic.agreement(entities=entities) == float(correct):
                self.report(mode)
                return generic

        return None
