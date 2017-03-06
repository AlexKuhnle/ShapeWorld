from shapeworld.world import World


class WorldCaptioner(object):

    def __init__(self, caption_size, words):
        self.caption_size = caption_size
        if words:
            words = sorted(words)
            self.word_ids = {words[n]: n + 1 for n in range(len(words))}
            self.word_ids[''] = 0
        else:
            self.word_ids = dict()

    def __call__(self, world, mode=None):
        assert isinstance(world, World)
        assert mode in (None, 'train', 'test', 'validation')
        if mode is None:
            captioner = self.caption_world
        elif mode == 'train':
            captioner = self.caption_train_world
        elif mode == 'test':
            captioner = self.caption_test_world
        elif mode == 'validation':
            captioner = self.caption_validation_world
        for _ in range(10):
            caption = captioner(world)
            if caption is not None:
                return caption
        assert False

    def caption_world(self, world):
        raise NotImplementedError

    def caption_train_world(self, world):
        return self.caption_world(world)

    def caption_validation_world(self, world):
        return self.caption_world(world)

    def caption_test_world(self, world):
        return self.caption_world(world)

    def realize(self, captions):
        return captions
