from importlib import import_module


class CaptionRealizer(object):

    modifiers = None
    relations = None
    quantifiers = None

    def __init__(self, language):
        self.language = language

    @staticmethod
    def from_name(name, language):
        assert isinstance(name, str)
        module = import_module('shapeworld.realizers.{}.realizer'.format(name))
        realizer_class = module.realizer
        realizer = realizer_class(language=language)
        return realizer

    def realize(self, captions):
        raise NotImplementedError
