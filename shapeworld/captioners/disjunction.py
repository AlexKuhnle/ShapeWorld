from random import choice
from shapeworld import util
from shapeworld.caption import Proposition
from shapeworld.captioners import WorldCaptioner


class DisjunctionCaptioner(WorldCaptioner):

    # modes
    # 0: incorrect (both incorrect)
    # 1: correct (first correct)
    # 2: correct (second correct)
    # 3: correct (both correct)

    name = 'disjunction'
    statistics_header = 'correct,captioner1,captioner2,mode'

    def __init__(self, captioners, correct_distribution=None):
        super(DisjunctionCaptioner, self).__init__()
        self.captioners = list(captioners)
        self.correct_distribution = util.cumulative_distribution(correct_distribution or [1, 1, 1])

    def set_realizer(self, realizer):
        if not super(DisjunctionCaptioner, self).set_realizer(realizer=realizer):
            return False
        assert 'disjunction' in realizer.propositions
        for captioner in self.captioners:
            captioner.set_realizer(realizer=realizer)
        return True

    def caption_world(self, entities, correct):
        captioner1 = choice(self.captioners)
        captioner2 = choice(self.captioners)

        if not correct:
            mode = 0
        else:
            mode = 1 + util.sample(self.correct_distribution)

        if (mode == 0) == correct:
            return None

        for _ in range(DisjunctionCaptioner.MAX_ATTEMPTS):

            if mode == 0:  # both incorrect
                clause1 = captioner1.caption_world(entities=entities, correct=False)
                clause2 = captioner2.caption_world(entities=entities, correct=False)

            elif mode == 1:  # first correct
                clause1 = captioner1.caption_world(entities=entities, correct=True)
                clause2 = captioner2.caption_world(entities=entities, correct=False)

            elif mode == 2:  # second correct
                clause1 = captioner1.caption_world(entities=entities, correct=False)
                clause2 = captioner2.caption_world(entities=entities, correct=True)

            elif mode == 3:  # both correct
                clause1 = captioner1.caption_world(entities=entities, correct=True)
                clause2 = captioner2.caption_world(entities=entities, correct=True)

            if clause1 is None or clause2 is None:
                continue

            proposition = Proposition(proptype='disjunction', clauses=(clause1, clause2))

            if proposition.agreement(entities=entities) == float(correct):
                self.report(correct, str(captioner1), str(captioner2), mode)
                return proposition

        return None
