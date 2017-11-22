from random import random
from shapeworld import util
from shapeworld.world import World
from shapeworld.captioners import LogicalPredication, PragmaticalPredication


class WorldCaptioner(object):

    MAX_SAMPLE_ATTEMPTS = 10
    MAX_ATTEMPTS = 4

    def __init__(self, internal_captioners, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        self.internal_captioners = list(internal_captioners)
        self.pragmatical_redundancy_rate = util.value_or_default(pragmatical_redundancy_rate, 1.0)
        self.pragmatical_tautology_rate = util.value_or_default(pragmatical_tautology_rate, 0.0)
        self.logical_redundancy_rate = util.value_or_default(logical_redundancy_rate, 1.0)
        self.logical_tautology_rate = util.value_or_default(logical_tautology_rate, 0.0)
        self.logical_contradiction_rate = util.value_or_default(logical_contradiction_rate, 0.0)
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

    def rpn_length(self):
        return max(captioner.rpn_length() for captioner in self.internal_captioners)

    def rpn_symbols(self):
        return set(rpn_symbol for captioner in self.internal_captioners for rpn_symbol in captioner.rpn_symbols())

    def initialize(self, mode, correct):
        return self.sample_values(mode=mode, correct=correct, predication=LogicalPredication())

    def sample_values(self, mode, correct, predication):
        assert mode in (None, 'train', 'validation', 'test')
        assert isinstance(correct, bool)
        assert isinstance(predication, LogicalPredication)

        self.mode = mode
        self.correct = correct

        if self.pragmatical_redundancy_rate == 0.0:
            self.pragmatical_redundancy = False
        elif self.pragmatical_redundancy_rate == 1.0:
            self.pragmatical_redundancy = True
        else:
            self.pragmatical_redundancy = random() < self.pragmatical_redundancy_rate

        if self.pragmatical_tautology_rate == 0.0:
            self.pragmatical_tautology = False
        elif self.pragmatical_tautology_rate == 1.0:
            self.pragmatical_tautology = True
        else:
            self.pragmatical_tautology = random() < self.pragmatical_tautology_rate

        if self.logical_redundancy_rate == 0.0:
            self.logical_redundancy = False
        elif self.logical_redundancy_rate == 1.0:
            self.logical_redundancy = True
        else:
            self.logical_redundancy = random() < self.logical_redundancy_rate

        if self.logical_tautology_rate == 0.0:
            self.logical_tautology = False
        elif self.logical_tautology_rate == 1.0:
            self.logical_tautology = True
        else:
            self.logical_tautology = random() < self.logical_tautology_rate

        if self.logical_contradiction_rate == 0.0:
            self.logical_contradiction = False
        elif self.logical_contradiction_rate == 1.0:
            self.logical_contradiction = True
        else:
            self.logical_contradiction = random() < self.logical_contradiction_rate

        return True

    def model(self):
        return dict(
            name=str(self),
            mode=self.mode,
            correct=self.correct,
            logical_tautology=self.logical_tautology,
            logical_redundancy=self.logical_redundancy,
            pragmatical_redundancy=self.pragmatical_redundancy
        )

    def __call__(self, world):
        assert self.realizer is not None
        assert isinstance(world, World)

        for _ in range(self.__class__.MAX_ATTEMPTS):
            predication = PragmaticalPredication(agreeing=world.entities)

            caption = self.caption(predication=predication, world=world)
            if caption is None:
                continue

            agreement = caption.agreement(predication=predication, world=world)
            if agreement > 0.0:
                break

        else:
            return None

        if not self.correct:
            for _ in range(self.__class__.MAX_ATTEMPTS):
                predication = PragmaticalPredication(agreeing=world.entities)

                if not self.incorrect(caption=caption, predication=predication, world=world):
                    continue

                agreement = caption.agreement(predication=predication, world=world)
                if agreement < 0.0:
                    break

            else:
                return None

        return caption

    def caption(self, predication, world):
        raise NotImplementedError

    def incorrect(self, caption, predication, world):
        raise NotImplementedError

    def apply_caption_to_predication(self, caption, predication):
        raise NotImplementedError


class CaptionerMixer(WorldCaptioner):

    def __init__(self, captioners, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None, pragmatical_redundancy_rate=None, pragmatical_tautology_rate=None, logical_redundancy_rate=None, logical_tautology_rate=None, logical_contradiction_rate=None):
        assert len(captioners) >= 1
        assert not distribution or len(distribution) == len(captioners)
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(distribution)
        super(CaptionerMixer, self).__init__(
            internal_captioners=captioners,
            pragmatical_redundancy_rate=pragmatical_redundancy_rate,
            pragmatical_tautology_rate=pragmatical_tautology_rate,
            logical_redundancy_rate=logical_redundancy_rate,
            logical_tautology_rate=logical_tautology_rate,
            logical_contradiction_rate=logical_contradiction_rate
        )
        distribution = util.value_or_default(distribution, [1] * len(captioners))
        self.distribution = util.cumulative_distribution(distribution)
        self.train_distribution = util.cumulative_distribution(util.value_or_default(train_distribution, distribution))
        self.validation_distribution = util.cumulative_distribution(util.value_or_default(validation_distribution, distribution))
        self.test_distribution = util.cumulative_distribution(util.value_or_default(test_distribution, distribution))

    def sample_values(self, mode, correct, predication):
        if not super(CaptionerMixer, self).sample_values(mode=mode, correct=correct, predication=predication):
            return False

        if mode is None:
            self.captioner = util.sample(self.distribution, self.internal_captioners)
        elif mode == 'train':
            self.captioner = util.sample(self.train_distribution, self.internal_captioners)
        elif mode == 'validation':
            self.captioner = util.sample(self.validation_distribution, self.internal_captioners)
        elif mode == 'test':
            self.captioner = util.sample(self.test_distribution, self.internal_captioners)

        return self.captioner.sample_values(mode=mode, correct=correct, predication=predication)

    def model(self):
        return util.merge_dicts(
            dict1=super(CaptionerMixer, self).model(),
            dict2=dict(captioner=self.captioner.model())
        )

    def caption(self, predication, world):
        return self.captioner.caption(predication=predication, world=world)

    def incorrect(self, caption, predication, world):
        return self.captioner.incorrect(caption=caption, predication=predication, world=world)

    def apply_caption_to_predication(self, caption, predication):
        return self.captioner.apply_caption_to_predication(caption=caption, predication=predication)
