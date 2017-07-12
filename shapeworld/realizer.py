from importlib import import_module
from shapeworld.util import powerset
from shapeworld.caption import Noun


class CaptionRealizer(object):

    modifiers = None
    relations = None
    quantifiers = None

    def __init__(self):
        assert self.__class__.modifiers and self.__class__.relations and self.__class__.quantifiers

    @staticmethod
    def from_name(name):
        assert isinstance(name, str)
        module = import_module('shapeworld.realizers.{}.realizer'.format(name))
        realizer_class = module.realizer
        realizer = realizer_class()
        return realizer

    def realize(self, captions):
        raise NotImplementedError

    def get_modifiers(self, modtypes=None, names=None):
        if names:
            assert not modtypes
            return [self.__class__.modifier_by_name[name] for name in names]
        elif modtypes:
            return [(modtype, value) for modtype in modtypes if modtype in self.__class__.modifiers for value in self.__class__.modifiers[modtype].keys()]
        else:
            return [(modtype, value) for modtype, modifiers in self.__class__.modifiers.items() for value in modifiers.keys()]

    def get_relations(self, reltypes=None, names=None):
        if names:
            assert not reltypes
            return [self.__class__.relations_by_name[name] for name in names]
        elif reltypes:
            return [(reltype, value) for reltype in reltypes if reltype in self.__class__.relations for value in self.__class__.relations[reltype].keys()]
        else:
            return [(reltype, value) for reltype, relations in self.__class__.relations.items() for value in relations.keys()]

    def get_quantifiers(self, qtypes=None, qranges=None, names=None):
        if names:
            assert not qtypes and not qranges
            return [self.__class__.quantifier_by_name[name] for name in names]
        elif qtypes and qranges:
            return [(qtype, qrange, value) for qtype in qtypes if qtype in self.__class__.quantifiers for qrange in qranges if qrange in self.__class__.quantifiers[qtype] for value in self.__class__.quantifiers[qtype][qrange].keys()]
        elif qtypes:
            return [(qtype, qrange, value) for qtype in qtypes if qtype in self.__class__.quantifiers for qrange, quantifiers in self.__class__.quantifiers[qtype].items() for value in quantifiers.keys()]
        elif qranges:
            return [(qtype, qrange, value) for qtype, quantifiers in self.__class__.quantifiers.items() for qrange in qranges if qrange in quantifiers for value in quantifiers[qrange].keys()]
        else:
            return [(qtype, qrange, value) for qtype, quantifiers1 in self.__class__.quantifiers.items() for qrange, quantifiers2 in quantifiers1.items() for value in quantifiers2.keys()]

    def noun_for_entity(self, entity):
        raise NotImplementedError

    def hypernyms_for_entity(self, entity, noun, include_universal=True):
        return [Noun(predicates=predicates) for predicates in powerset(values=noun.predicates, min_num=(0 if include_universal else 1), max_num=(len(noun.predicates) - 1))]

    # def hyponyms_for_entity(self, entity, noun):
    #     return ()
