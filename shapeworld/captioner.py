import os
from shapeworld.util import cumulative_distribution, sample
from shapeworld.world import World


class WorldCaptioner(object):

    MAX_ATTEMPTS = 10
    name = None
    statistics_header = None

    def __init__(self, quantifier_tolerance=None):
        assert self.__class__.name
        self.quantifier_tolerance = quantifier_tolerance if quantifier_tolerance is not None else 0.1
        self.statistics_filehandle = None
        self.realizer = None

    def __str__(self):
        return self.__class__.name

    def set_realizer(self, realizer):
        if self.realizer is None:
            self.realizer = realizer
            return True
        else:
            return False

    def __call__(self, world, correct, mode=None):
        assert self.realizer
        assert isinstance(world, dict) or isinstance(world, World)
        assert isinstance(correct, bool)
        assert mode in (None, 'train', 'test', 'validation')
        if isinstance(world, World):
            world = world.model()
        if mode is None:
            captioner = self.caption_world
        elif mode == 'train':
            captioner = self.caption_train_world
        elif mode == 'test':
            captioner = self.caption_test_world
        elif mode == 'validation':
            captioner = self.caption_validation_world
        for _ in range(WorldCaptioner.MAX_ATTEMPTS):
            caption = captioner(world, correct)
            if caption is not None:
                return caption
        return None

    def caption_world(self, world, correct):
        raise NotImplementedError

    def caption_train_world(self, world, correct):
        return self.caption_world(world, correct)

    def caption_validation_world(self, world, correct):
        return self.caption_world(world, correct)

    def caption_test_world(self, world, correct):
        return self.caption_world(world, correct)

    def collect_statistics(self, path, append=False):
        assert isinstance(path, str)
        self.statistics_filehandle = open(path, 'a' if append else 'w')
        if not append and self.__class__.statistics_header:
            self.statistics_filehandle.write(self.__class__.statistics_header + '\n')

    def close_statistics(self):
        if self.statistics_filehandle is not None:
            self.statistics_filehandle.close()

    def report(self, *instance):
        if self.statistics_filehandle is not None:
            self.statistics_filehandle.write(','.join(str(value) for value in instance) + '\n')


class CaptionerMixer(WorldCaptioner):

    name = 'mixer'
    statistics_header = 'various'

    def __init__(self, captioners, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        assert len(captioners) >= 1
        assert all(captioner.quantifier_tolerance == captioners[0].quantifier_tolerance for captioner in captioners)
        self.captioners = captioners
        super(CaptionerMixer, self).__init__(quantifier_tolerance=captioners[0].quantifier_tolerance)
        assert not distribution or len(distribution) == len(captioners)
        self.distribution = cumulative_distribution(distribution or [1] * len(captioners))
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(self.distribution)
        self.train_distribution = cumulative_distribution(train_distribution) if train_distribution else self.distribution
        self.validation_distribution = cumulative_distribution(validation_distribution) if validation_distribution else self.distribution
        self.test_distribution = cumulative_distribution(test_distribution) if test_distribution else self.distribution

    def set_realizer(self, realizer):
        if super(CaptionerMixer, self).set_realizer(realizer=realizer):
            for captioner in self.captioners:
                captioner.set_realizer(self.realizer)
            return True
        else:
            return False

    def __call__(self, world, correct, mode=None):
        assert self.realizer
        assert isinstance(world, dict) or isinstance(world, World)
        assert isinstance(correct, bool)
        assert mode in (None, 'train', 'test', 'validation')
        if isinstance(world, World):
            world = world.model()
        if mode is None:
            captioner = sample(self.distribution, self.captioners)
            captioner = captioner.caption_world
        elif mode == 'train':
            captioner = sample(self.train_distribution, self.captioners)
            captioner = captioner.caption_train_world
        elif mode == 'validation':
            captioner = sample(self.validation_distribution, self.captioners)
            captioner = captioner.caption_validation_world
        elif mode == 'test':
            captioner = sample(self.test_distribution, self.captioners)
            captioner = captioner.caption_test_world
        for _ in range(WorldCaptioner.MAX_ATTEMPTS):
            caption = captioner(world, correct)
            if caption is not None:
                return caption
        return None

    def caption_world(self, world, correct):
        captioner = sample(self.distribution, self.captioners)
        raise captioner.caption_world(world, correct)

    def caption_train_world(self, world, correct):
        captioner = sample(self.train_distribution, self.captioners)
        return captioner.caption_world(world, correct)

    def caption_validation_world(self, world, correct):
        captioner = sample(self.validation_distribution, self.captioners)
        return captioner.caption_world(world, correct)

    def caption_test_world(self, world, correct):
        captioner = sample(self.test_distribution, self.captioners)
        return captioner.caption_world(world, correct)

    def collect_statistics(self, path, append=False):
        super(CaptionerMixer, self).collect_statistics(path=path, append=append)
        for captioner in self.captioners:
            captioner.statistics_filehandle = self.statistics_filehandle
