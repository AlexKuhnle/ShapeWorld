from random import choice
from shapeworld import util
from shapeworld.caption import Proposition
from shapeworld.captioners import WorldCaptioner


class DisjunctionCaptioner(WorldCaptioner):

    # correct modes
    # 0: incorrect (both incorrect)
    # 1: correct (first correct)
    # 2: correct (second correct)
    # 3: correct (both correct)

    def __init__(self, captioners, correct_distribution=None, trivial_acceptance_rate=None):
        super(DisjunctionCaptioner, self).__init__(internal_captioners=captioners, trivial_acceptance_rate=trivial_acceptance_rate)
        self.correct_distribution = util.cumulative_distribution(util.value_or_default(correct_distribution, [1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(DisjunctionCaptioner, self).set_realizer(realizer=realizer):
            return False
        assert 'disjunction' in realizer.propositions
        return True

    def sample_values(self, mode, correct):
        super(DisjunctionCaptioner, self).sample_values(mode=mode, correct=correct)

        self.captioner1 = choice(self.internal_captioners)
        self.captioner2 = choice(self.internal_captioners)
        self.correct_mode = 0 if not correct else 1 + util.sample(self.correct_distribution)

        if self.correct_mode == 0:  # both incorrect
            self.captioner1.sample_values(mode=mode, correct=False)
            self.captioner2.sample_values(mode=mode, correct=False)
        elif self.correct_mode == 1:  # first correct
            self.captioner1.sample_values(mode=mode, correct=True)
            self.captioner2.sample_values(mode=mode, correct=False)
        elif self.correct_mode == 2:  # second correct
            self.captioner1.sample_values(mode=mode, correct=False)
            self.captioner2.sample_values(mode=mode, correct=True)
        elif self.correct_mode == 3:  # both correct
            self.captioner1.sample_values(mode=mode, correct=True)
            self.captioner2.sample_values(mode=mode, correct=True)

    def model(self):
        return util.merge_dicts(
            dict1=super(DisjunctionCaptioner, self).model(),
            dict2=dict(correct_mode=self.correct_mode, captioner1=self.captioner1.model(), captioner2=self.captioner2.model())
        )

    def caption_world(self, entities, relevant_entities):
        clause1 = self.captioner1.caption_world(entities=entities, relevant_entities=relevant_entities)
        if clause1 is None:
            return None

        clause2 = self.captioner2.caption_world(entities=entities, relevant_entities=relevant_entities)
        if clause2 is None:
            return None

        return Proposition(proptype='disjunction', clauses=(clause1, clause2))
