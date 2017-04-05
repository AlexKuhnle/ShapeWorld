from shapeworld import CaptionRealizer
from shapeworld.world import World


class WorldCaptioner(object):

    MAX_ATTEMPTS = 10
    statistics_header = None

    def __init__(self, realizer=None, quantifier_tolerance=None):
        if realizer is None:
            realizer = CaptionRealizer.from_name(name=(realizer or 'dmrs'))
        assert isinstance(realizer, CaptionRealizer)
        self.realizer = realizer
        self.quantifier_tolerance = quantifier_tolerance if quantifier_tolerance is not None else 0.1
        self.statistics_filehandle = None

    def __call__(self, world, correct, mode=None):
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
        assert False

    def caption_world(self, world, correct):
        raise NotImplementedError

    def caption_train_world(self, world, correct):
        return self.caption_world(world, correct)

    def caption_validation_world(self, world, correct):
        return self.caption_world(world, correct)

    def caption_test_world(self, world, correct):
        return self.caption_world(world, correct)

    def realize(self, captions):
        return self.realizer.realize(captions=captions)

    def collect_statistics(self, filehandle, append=False):
        assert filehandle is not None
        self.statistics_filehandle = filehandle
        if not append and self.__class__.statistics_header:
            self.statistics_filehandle.write(self.__class__.statistics_header + '\n')

    def close_statistics(self):
        if self.statistics_filehandle is not None:
            self.statistics_filehandle.close()

    def report(self, *instance):
        if self.statistics_filehandle is not None:
            self.statistics_filehandle.write(','.join(str(value) for value in instance) + '\n')
