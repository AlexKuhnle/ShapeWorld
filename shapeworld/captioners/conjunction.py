from random import choice
from shapeworld import util
from shapeworld.caption import Proposition
from shapeworld.captioners import WorldCaptioner


class ConjunctionCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct (both correct)
    # 1: incorrect (second incorrect)
    # 2: incorrect (first incorrect)
    # 3: incorrect (both incorrect)

    def __init__(self, captioners, incorrect_distribution=None, trivial_acceptance_rate=None):
        super(ConjunctionCaptioner, self).__init__(internal_captioners=captioners, trivial_acceptance_rate=trivial_acceptance_rate)
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1, 1]))

    def set_realizer(self, realizer):
        if not super(ConjunctionCaptioner, self).set_realizer(realizer=realizer):
            return False
        assert 'conjunction' in realizer.propositions
        return True

    def sample_values(self, mode, correct):
        super(ConjunctionCaptioner, self).sample_values(mode=mode, correct=correct)

        self.captioner1 = choice(self.internal_captioners)
        self.captioner2 = choice(self.internal_captioners)
        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        if self.incorrect_mode == 0:  # both correct
            self.captioner1.sample_values(mode=mode, correct=True)
            self.captioner2.sample_values(mode=mode, correct=True)
        elif self.incorrect_mode == 1:  # second incorrect
            self.captioner1.sample_values(mode=mode, correct=True)
            self.captioner2.sample_values(mode=mode, correct=False)
        elif self.incorrect_mode == 2:  # first incorrect
            self.captioner1.sample_values(mode=mode, correct=False)
            self.captioner2.sample_values(mode=mode, correct=True)
        elif self.incorrect_mode == 3:  # both incorrect
            self.captioner1.sample_values(mode=mode, correct=False)
            self.captioner2.sample_values(mode=mode, correct=False)

    def model(self):
        return util.merge_dicts(
            dict1=super(ConjunctionCaptioner, self).model(),
            dict2=dict(incorrect_mode=self.incorrect_mode, captioner1=self.captioner1.model(), captioner2=self.captioner2.model())
        )

    def caption_world(self, entities, relevant_entities):
        clause1 = self.captioner1.caption_world(entities=entities, relevant_entities=relevant_entities)
        if clause1 is None:
            return None

        clause2 = self.captioner2.caption_world(entities=entities, relevant_entities=relevant_entities)
        if clause2 is None:
            return None

        return Proposition(proptype='conjunction', clauses=(clause1, clause2))
