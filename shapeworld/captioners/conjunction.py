from random import choice
from shapeworld import util
from shapeworld.caption import Proposition
from shapeworld.captioners import WorldCaptioner


class ConjunctionCaptioner(WorldCaptioner):

    # modes
    # 0: correct (both correct)
    # 1: incorrect (first incorrect)
    # 2: incorrect (second incorrect)
    # 3: incorrect (both incorrect)

    name = 'conjunction'
    statistics_header = 'correct,captioner1,captioner2,mode'

    def __init__(self, captioners, incorrect_distribution=None):
        super(ConjunctionCaptioner, self).__init__()
        self.captioners = list(captioners)
        self.incorrect_distribution = util.cumulative_distribution(incorrect_distribution or [1, 1, 1])

    def set_realizer(self, realizer):
        if not super(ConjunctionCaptioner, self).set_realizer(realizer=realizer):
            return False
        assert 'conjunction' in realizer.propositions
        for captioner in self.captioners:
            captioner.set_realizer(realizer=realizer)
        return True

    def caption_world(self, entities, correct):
        captioner1 = choice(self.captioners)
        captioner2 = choice(self.captioners)

        if correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.incorrect_distribution)

        if (mode == 0) != correct:
            return None

        for _ in range(ConjunctionCaptioner.MAX_ATTEMPTS):

            if mode == 0:  # both correct
                clause1 = captioner1.caption_world(entities=entities, correct=True)
                clause2 = captioner2.caption_world(entities=entities, correct=True)

            elif mode == 1:  # first incorrect
                clause1 = captioner1.caption_world(entities=entities, correct=False)
                clause2 = captioner2.caption_world(entities=entities, correct=True)

            elif mode == 2:  # second incorrect
                clause1 = captioner1.caption_world(entities=entities, correct=True)
                clause2 = captioner2.caption_world(entities=entities, correct=False)

            elif mode == 3:  # both incorrect
                clause1 = captioner1.caption_world(entities=entities, correct=False)
                clause2 = captioner2.caption_world(entities=entities, correct=False)

            if clause1 is None or clause2 is None:
                continue

            proposition = Proposition(proptype='conjunction', clauses=(clause1, clause2))

            if proposition.agreement(entities=entities) == float(correct):
                self.report(correct, str(captioner1), str(captioner2), mode)
                return proposition

        return None
