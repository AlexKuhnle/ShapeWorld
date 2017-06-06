from random import choice, random
from shapeworld import WorldCaptioner
from shapeworld.util import cumulative_distribution
from shapeworld.caption import Proposition


class ConjunctionCaptioner(WorldCaptioner):

    name = 'conjunction'
    statistics_header = 'correct,mode,captioner1,captioner2'

    def __init__(self, captioners, quantifier_tolerance=None, incorrect_distribution=None):
        # requires connective 'conjunction'
        super(ConjunctionCaptioner, self).__init__(quantifier_tolerance=quantifier_tolerance)
        self.captioners = list(captioners)
        self.incorrect_distribution = cumulative_distribution(incorrect_distribution or [1, 1, 1])

    def set_realizer(self, realizer):
        if super(ConjunctionCaptioner, self).set_realizer(realizer=realizer):
            for captioner in self.captioners:
                captioner.set_realizer(realizer=self.realizer)
            return True
        else:
            return False

    def caption_world(self, world, correct):
        mode = 0
        for _ in range(ConjunctionCaptioner.MAX_ATTEMPTS):
            captioner1 = choice(self.captioners)
            captioner2 = choice(self.captioners)

            if correct:
                proposition1 = captioner1.caption_world(world=world, correct=True)
                proposition2 = captioner2.caption_world(world=world, correct=True)

            else:
                r = random()
                if r < self.incorrect_distribution[0]:  # first incorrect
                    mode = 1
                    proposition1 = captioner1.caption_world(world=world, correct=False)
                    proposition2 = captioner2.caption_world(world=world, correct=True)
                elif r < self.incorrect_distribution[1]:  # second incorrect
                    mode = 2
                    proposition1 = captioner1.caption_world(world=world, correct=True)
                    proposition2 = captioner2.caption_world(world=world, correct=False)
                elif r < self.incorrect_distribution[2]:  # both incorrect
                    mode = 3
                    proposition1 = captioner1.caption_world(world=world, correct=False)
                    proposition2 = captioner2.caption_world(world=world, correct=False)

            if not proposition1 or not proposition2:
                continue
            assert len(proposition1.clauses) == 1
            assert len(proposition2.clauses) == 1

            caption = Proposition(clauses=(proposition1.clauses[0], proposition2.clauses[0]), connective='conjunction')
            if caption.agreement(world=world) == float(correct):
                self.report(correct, mode, str(captioner1), str(captioner2))
                return caption

        return None
