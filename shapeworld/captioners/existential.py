from shapeworld import util
from shapeworld.caption import Existential
from shapeworld.captioners import WorldCaptioner, AttributesTypeCaptioner


class ExistentialCaptioner(WorldCaptioner):

    # incorrect modes
    # 0: correct
    # 1: incorrect restrictor
    # 2: incorrect body

    def __init__(self, restrictor_captioner=None, body_captioner=None, incorrect_distribution=None, trivial_acceptance_rate=None):
        self.restrictor_captioner = util.value_or_default(restrictor_captioner, AttributesTypeCaptioner())
        self.body_captioner = util.value_or_default(body_captioner, AttributesTypeCaptioner())
        super(ExistentialCaptioner, self).__init__(internal_captioners=(self.restrictor_captioner, self.body_captioner), trivial_acceptance_rate=trivial_acceptance_rate)
        self.incorrect_distribution = util.cumulative_distribution(util.value_or_default(incorrect_distribution, [1, 1]))

    def sample_values(self, mode, correct):
        super(ExistentialCaptioner, self).sample_values(mode=mode, correct=correct)

        self.incorrect_mode = 0 if correct else 1 + util.sample(self.incorrect_distribution)

        self.restrictor_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 1))  # 1: incorrect restrictor
        self.body_captioner.sample_values(mode=mode, correct=(self.incorrect_mode != 2))  # 2: incorrect body

    def model(self):
        return util.merge_dicts(
            dict1=super(ExistentialCaptioner, self).model(),
            dict2=dict(incorrect_mode=self.incorrect_mode, restrictor_captioner=self.restrictor_captioner.model(), body_captioner=self.body_captioner.model())
        )

    def caption_world(self, entities, relevant_entities):
        body = self.body_captioner(entities=entities, relevant_entities=relevant_entities)
        if body is None:
            return None

        if self.incorrect_mode == 2:  # 2: incorrect body
            relevant_entities_restrictor = body.disagreeing_entities(entities=relevant_entities)
        else:
            relevant_entities_restrictor = body.agreeing_entities(entities=relevant_entities)

        restrictor = self.restrictor_captioner(entities=entities, relevant_entities=relevant_entities_restrictor)
        if restrictor is None:
            return None

        return Existential(restrictor=restrictor, body=body)
