from random import choice, random
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Proposition


class DisjunctionCaptioner(WorldCaptioner):

    name = 'disjunction'
    statistics_header = 'correct,mode,captioner1,captioner2'

    def __init__(self, captioners, quantifier_tolerance=None, correct_distribution=None):
        # requires connective 'conjunction'
        super(DisjunctionCaptioner, self).__init__(quantifier_tolerance=quantifier_tolerance)
        self.captioners = list(captioners)
        self.correct_distribution = cumulative_distribution(correct_distribution or [1, 1, 1])

    def set_realizer(self, realizer):
        if super(DisjunctionCaptioner, self).set_realizer(realizer=realizer):
            for captioner in self.captioners:
                captioner.set_realizer(realizer=self.realizer)
            return True
        else:
            return False

    def caption_world(self, world, correct):
        mode = 0
        for _ in range(DisjunctionCaptioner.MAX_ATTEMPTS):
            captioner1 = choice(self.captioners)
            captioner2 = choice(self.captioners)

            if correct:
                r = random()
                if r < self.correct_distribution[0]:  # first correct
                    mode = 1
                    proposition1 = captioner1.caption_world(world=world, correct=True)
                    proposition2 = captioner2.caption_world(world=world, correct=False)
                elif r < self.correct_distribution[1]:  # second correct
                    mode = 2
                    proposition1 = captioner1.caption_world(world=world, correct=False)
                    proposition2 = captioner2.caption_world(world=world, correct=True)
                elif r < self.correct_distribution[2]:  # both correct
                    mode = 3
                    proposition1 = captioner1.caption_world(world=world, correct=True)
                    proposition2 = captioner2.caption_world(world=world, correct=True)

            else:
                proposition1 = captioner1.caption_world(world=world, correct=False)
                proposition2 = captioner2.caption_world(world=world, correct=False)

            if not proposition1 or not proposition2:
                continue
            assert len(proposition1.clauses) == 1
            assert len(proposition2.clauses) == 1

            caption = Proposition(clauses=(proposition1.clauses[0], proposition2.clauses[0]), connective='disjunction')
            if caption.agreement(world=world) == float(correct):
                self.report(correct, mode, str(captioner1), str(captioner2))
                return caption

        return None
