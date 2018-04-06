from importlib import import_module
from shapeworld import util


class CaptionRealizer(object):

    def __init__(self, language):
        self.language = language
        self.attributes = None
        self.relations = None
        self.quantifiers = None

    @staticmethod
    def from_name(name, language):
        assert isinstance(name, str)
        module = import_module('shapeworld.realizers.' + name)
        class_name = util.class_name(name) + 'Realizer'
        for key, module in module.__dict__.items():
            if key == class_name:
                break
        realizer = module(language=language)
        return realizer

    def realize(self, captions):
        raise NotImplementedError
