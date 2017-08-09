from importlib import import_module
from shapeworld.util import powerset
from shapeworld.caption import Noun


class CaptionRealizer(object):

    modifiers = None
    relations = None
    quantifiers = None

    @staticmethod
    def from_name(name, language):
        assert isinstance(name, str)
        module = import_module('shapeworld.realizers.{}.realizer'.format(name))
        realizer_class = module.realizer
        realizer = realizer_class(language=language)
        return realizer

    def realize(self, captions):
        raise NotImplementedError

    def get_modifiers(self, modtypes=None, names=None):
        if names:
            assert not modtypes
            return [self.modifier_by_name[name] for name in names]
        elif modtypes:
            return [(modtype, value) for modtype in modtypes if modtype in self.modifiers for value in self.modifiers[modtype].keys()]
        else:
            return [(modtype, value) for modtype, modifiers in self.modifiers.items() for value in modifiers.keys()]

    def get_relations(self, reltypes=None, names=None):
        if names:
            assert not reltypes
            return [self.relations_by_name[name] for name in names]
        elif reltypes:
            return [(reltype, value) for reltype in reltypes if reltype in self.relations for value in self.relations[reltype].keys()]
        else:
            return [(reltype, value) for reltype, relations in self.relations.items() for value in relations.keys()]

    def get_quantifiers(self, qtypes=None, qranges=None, names=None):
        if names:
            assert not qtypes and not qranges
            return [self.quantifier_by_name[name] for name in names]
        elif qtypes and qranges:
            return [(qtype, qrange, value) for qtype in qtypes if qtype in self.quantifiers for qrange in qranges if qrange in self.quantifiers[qtype] for value in self.quantifiers[qtype][qrange].keys()]
        elif qtypes:
            return [(qtype, qrange, value) for qtype in qtypes if qtype in self.quantifiers for qrange, quantifiers in self.quantifiers[qtype].items() for value in quantifiers.keys()]
        elif qranges:
            return [(qtype, qrange, value) for qtype, quantifiers in self.quantifiers.items() for qrange in qranges if qrange in quantifiers for value in quantifiers[qrange].keys()]
        else:
            return [(qtype, qrange, value) for qtype, quantifiers1 in self.quantifiers.items() for qrange, quantifiers2 in quantifiers1.items() for value in quantifiers2.keys()]
