from shapeworld.util import cumulative_distribution, sample


class WorldCaptioner(object):

    MAX_ATTEMPTS = 10
    name = None
    statistics_header = None

    def __init__(self):
        assert self.__class__.name
        self.statistics_filehandle = None
        self.realizer = None

    def __str__(self):
        return self.__class__.name

    def set_realizer(self, realizer):
        self.realizer = realizer

    def __call__(self, entities, correct, mode=None):
        assert self.realizer
        assert isinstance(entities, list)
        assert isinstance(correct, bool)
        assert mode in (None, 'train', 'test', 'validation')
        if mode is None:
            captioner = self.caption_world
        elif mode == 'train':
            captioner = self.caption_train_world
        elif mode == 'test':
            captioner = self.caption_test_world
        elif mode == 'validation':
            captioner = self.caption_validation_world
        for _ in range(WorldCaptioner.MAX_ATTEMPTS):
            caption = captioner(entities=entities, correct=correct)
            if caption is not None:
                return caption
        return None

    def caption_world(self, entities, correct):
        raise NotImplementedError

    def caption_train_world(self, entities, correct):
        return self.caption_world(entities, correct)

    def caption_validation_world(self, entities, correct):
        return self.caption_world(entities, correct)

    def caption_test_world(self, entities, correct):
        return self.caption_world(entities, correct)

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
        assert not distribution or len(distribution) == len(captioners)
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(self.distribution)
        super(CaptionerMixer, self).__init__()
        self.captioners = captioners
        self.distribution = cumulative_distribution(distribution or [1] * len(captioners))
        self.train_distribution = cumulative_distribution(train_distribution) if train_distribution else self.distribution
        self.validation_distribution = cumulative_distribution(validation_distribution) if validation_distribution else self.distribution
        self.test_distribution = cumulative_distribution(test_distribution) if test_distribution else self.distribution

    def set_realizer(self, realizer):
        super(CaptionerMixer, self).set_realizer(realizer=realizer)
        for captioner in self.captioners:
            captioner.set_realizer(self.realizer)

    def __call__(self, entities, correct, mode=None):
        assert self.realizer
        assert isinstance(entities, list)
        assert isinstance(correct, bool)
        assert mode in (None, 'train', 'test', 'validation')
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
            caption = captioner(entities=entities, correct=correct)
            if caption is not None:
                return caption
        return None

    def caption_world(self, entities, correct):
        captioner = sample(self.distribution, self.captioners)
        raise captioner.caption_world(entities, correct)

    def caption_train_world(self, entities, correct):
        captioner = sample(self.train_distribution, self.captioners)
        return captioner.caption_world(entities, correct)

    def caption_validation_world(self, entities, correct):
        captioner = sample(self.validation_distribution, self.captioners)
        return captioner.caption_world(entities, correct)

    def caption_test_world(self, entities, correct):
        captioner = sample(self.test_distribution, self.captioners)
        return captioner.caption_world(entities, correct)

    def collect_statistics(self, path, append=False):
        super(CaptionerMixer, self).collect_statistics(path=path, append=append)
        for captioner in self.captioners:
            captioner.statistics_filehandle = self.statistics_filehandle
