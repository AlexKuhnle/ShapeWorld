from random import random
from shapeworld import util


class WorldCaptioner(object):

    MAX_ATTEMPTS = 3

    def __init__(self, internal_captioners=None, trivial_acceptance_rate=None):
        self.internal_captioners = list(util.value_or_default(internal_captioners, ()))
        self.trivial_acceptance_rate = trivial_acceptance_rate
        if trivial_acceptance_rate is not None:
            captioners = list(self.internal_captioners)
            while captioners:
                captioner = captioners.pop()
                if captioner.trivial_acceptance_rate is None:
                    captioner.trivial_acceptance_rate = trivial_acceptance_rate
                    captioners.extend(captioner.internal_captioners)
        self.realizer = None
        self.correct = None

    def __str__(self):
        return self.__class__.__name__

    def set_realizer(self, realizer):
        if self.realizer is not None:
            return False
        self.realizer = realizer
        for captioner in self.internal_captioners:
            captioner.set_realizer(realizer)
        return True

    def sample_values(self, mode, correct):
        assert mode in (None, 'train', 'validation', 'test')
        assert isinstance(correct, bool)
        self.mode = mode
        self.correct = correct
        if self.trivial_acceptance_rate is None or self.trivial_acceptance_rate == 0.0:
            self.trivial_accepted = False
        elif self.trivial_acceptance_rate == 1.0:
            self.trivial_accepted = True
        else:
            self.trivial_accepted = random() < self.trivial_acceptance_rate

    def model(self):
        return dict(name=str(self), mode=self.mode, correct=self.correct, trivial_accepted=self.trivial_accepted)

    def __call__(self, entities, relevant_entities=None):
        assert self.realizer is not None
        assert isinstance(entities, list)

        if self.mode is None:
            captioner = self.caption_world
        elif self.mode == 'train':
            captioner = self.caption_train_world
        elif self.mode == 'validation':
            captioner = self.caption_validation_world
        elif self.mode == 'test':
            captioner = self.caption_test_world

        if relevant_entities is None:
            relevant_entities = entities

        for _ in range(self.__class__.MAX_ATTEMPTS):
            caption = captioner(entities=entities, relevant_entities=relevant_entities)
            if caption is None:
                continue
            agreement = caption.agreement(entities=relevant_entities)
            if agreement == 0.0:
                continue
            elif ((agreement == 2.0 and self.correct) or (agreement == -2.0 and not self.correct)) and not self.trivial_accepted:
                continue
            elif (agreement > 0.0 and self.correct) or (agreement < 0.0 and not self.correct):
                return caption
        return None

    def caption_world(self, entities, relevant_entities):
        raise NotImplementedError

    def caption_train_world(self, entities, relevant_entities):
        return self.caption_world(entities=entities, relevant_entities=relevant_entities)

    def caption_validation_world(self, entities, relevant_entities):
        return self.caption_train_world(entities=entities, relevant_entities=relevant_entities)

    def caption_test_world(self, entities, relevant_entities):
        return self.caption_world(entities=entities, relevant_entities=relevant_entities)


class CaptionerMixer(WorldCaptioner):

    def __init__(self, captioners, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None, trivial_acceptance_rate=None):
        assert len(captioners) >= 1
        assert not distribution or len(distribution) == len(captioners)
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(distribution)
        super(CaptionerMixer, self).__init__(internal_captioners=captioners, trivial_acceptance_rate=trivial_acceptance_rate)
        distribution = util.value_or_default(distribution, [1] * len(captioners))
        self.distribution = util.cumulative_distribution(distribution)
        self.train_distribution = util.cumulative_distribution(util.value_or_default(train_distribution, distribution))
        self.validation_distribution = util.cumulative_distribution(util.value_or_default(validation_distribution, distribution))
        self.test_distribution = util.cumulative_distribution(util.value_or_default(test_distribution, distribution))

    def sample_values(self, mode, correct):
        super(CaptionerMixer, self).sample_values(mode=mode, correct=correct)

        if mode is None:
            self.captioner = util.sample(self.distribution, self.internal_captioners)
        elif mode == 'train':
            self.captioner = util.sample(self.train_distribution, self.internal_captioners)
        elif mode == 'validation':
            self.captioner = util.sample(self.validation_distribution, self.internal_captioners)
        elif mode == 'test':
            self.captioner = util.sample(self.test_distribution, self.internal_captioners)

        self.captioner.sample_values(mode=mode, correct=correct)

    def model(self):
        return util.merge_dicts(
            dict1=super(CaptionerMixer, self).model(),
            dict2=dict(captioner=self.captioner.model())
        )

    def caption_world(self, entities, relevant_entities):
        return self.captioner.caption_world(entities=entities, relevant_entities=relevant_entities)

    def caption_train_world(self, entities, relevant_entities):
        return self.captioner.caption_train_world(entities=entities, relevant_entities=relevant_entities)

    def caption_validation_world(self, entities, relevant_entities):
        return self.captioner.caption_validation_world(entities=entities, relevant_entities=relevant_entities)

    def caption_test_world(self, entities, relevant_entities):
        return self.captioner.caption_test_world(entities=entities, relevant_entities=relevant_entities)
