import os
import re
import subprocess
from shapeworld import CaptionRealizer
from shapeworld.caption import Predicate, Modifier, Noun, Relation, Quantifier, Proposition
from shapeworld.realizers.dmrs.dmrs import ComposableDmrs


class DmrsRealizer(CaptionRealizer):

    def __init__(self):
        super(DmrsRealizer, self).__init__()
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
        self.ace_path = os.path.join(directory, 'ace')
        self.erg_path = os.path.join(directory, 'erg-shapeworld.dat')
        self.regex = re.compile(pattern=r'(NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\])|(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)')

    def realize(self, captions):
        assert all(isinstance(caption, Proposition) for caption in captions)
        ace = subprocess.Popen([self.ace_path, '-g', self.erg_path, '-1Te'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mrs = '\n'.join(self.proposition_dmrs(caption).get_mrs() for caption in captions) + '\n'
        stdout_data, stderr_data = ace.communicate(mrs.encode())
        # for n, line in enumerate(stderr_data.decode('utf-8').splitlines()):
        #     if not self.regex.match(line):
        #         print(line)
        #         print(self.proposition_dmrs(captions[n]).get_mrs())
        assert all(self.regex.match(line) for line in stderr_data.decode('utf-8').splitlines())
        caption_strings = stdout_data.decode('utf-8').splitlines()
        assert len(caption_strings) == len(captions)
        for n, caption in enumerate(caption_strings):
            caption = caption.lower()
            caption = caption.replace('.', ' .')
            captions[n] = caption.split()
        return captions

    def noun_for_entity(self, entity):
        return Noun(predicates=[
            Modifier(modtype='shape', value=entity['shape']['name']),
            Modifier(modtype='color', value=entity['color']['name'])])
        # Modifier(modtype='texture', value=entity['texture']['name'])])

    def clause_dmrs(self, clause):
        if isinstance(clause, Predicate):
            return self.predicate_dmrs(clause)
        elif isinstance(clause, Quantifier):
            return self.quantifier_dmrs(clause)
        elif isinstance(clause, Proposition):
            return self.proposition_dmrs(clause)
        else:
            assert False

    def predicate_dmrs(self, predicate):
        if isinstance(predicate, Modifier):
            return self.modifier_dmrs(predicate)
        elif isinstance(predicate, Relation):
            return self.relation_dmrs(predicate)
        elif isinstance(predicate, Noun):
            return self.noun_dmrs(predicate)
        else:
            assert False

    def modifier_dmrs(self, modifier):
        assert modifier.modtype in DmrsRealizer.modifiers and modifier.value in DmrsRealizer.modifiers[modifier.modtype]
        dmrs = DmrsRealizer.modifiers[modifier.modtype][modifier.value]
        tag = 'noun' if DmrsRealizer.modifier_is_noun[modifier.modtype] else 'modifier'
        return tag, dmrs

    def relation_dmrs(self, relation, require_plural=None):
        assert relation.reltype in DmrsRealizer.relations
        dmrs = DmrsRealizer.relations[relation.reltype]
        tag, ref_dmrs = self.predicate_dmrs(relation.reference)
        assert tag in ('modifier', 'noun')
        dmrs = dmrs.compose(ref_dmrs, {'ref': 'noun'})
        if tag == 'modifier':
            dmrs = dmrs.compose(DmrsRealizer.empty_predicate, {'ref': 'noun'})
            dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'ref': 'rstr'})
        elif tag == 'noun':
            dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'ref': 'rstr'})
        return 'relation', dmrs

    def noun_dmrs(self, noun):
        if not noun.predicates:
            return 'noun', DmrsRealizer.empty_predicate
        tag, dmrs = self.predicate_dmrs(noun.predicates[0])
        assert tag in ('modifier', 'noun')
        is_noun = tag == 'noun'
        for predicate in noun.predicates[1:]:
            tag, pred_dmrs = self.predicate_dmrs(predicate)
            assert tag in ('modifier', 'noun')
            dmrs = dmrs.compose(pred_dmrs, {'noun': 'noun'})
            if tag == 'noun':
                assert not is_noun
                is_noun = True
        tag = 'noun' if is_noun else 'modifier'
        return tag, dmrs

    def quantifier_dmrs(self, quantifier):
        tag, dmrs = self.predicate_dmrs(quantifier.body)
        if tag == 'modifier':
            dmrs = DmrsRealizer.adjective_relation.compose(dmrs, {'rel': 'mod'})
        elif tag == 'noun':
            dmrs = DmrsRealizer.empty_relation.compose(dmrs, {'ref': 'noun'})
            require_plural = (quantifier.qtype, quantifier.qrange, quantifier.quantity) in DmrsRealizer.quantifier_requires_plural
            if require_plural:
                dmrs = dmrs.compose(DmrsRealizer.empty_plural_quantifier, {'ref': 'rstr'})
            else:
                dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'ref': 'rstr'})
        tag, rstr_dmrs = self.predicate_dmrs(quantifier.restrictor)
        assert tag in ('modifier', 'noun')
        dmrs = dmrs.compose(rstr_dmrs, {'noun': 'noun'})
        if tag == 'modifier':
            dmrs = dmrs.compose(DmrsRealizer.empty_predicate, {'noun': 'noun'})
        dmrs = dmrs.compose(DmrsRealizer.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity], {'noun': 'rstr'})
        return 'quantifier', dmrs

    def proposition_dmrs(self, proposition):
        assert len(proposition.clauses) >= 1
        if len(proposition.clauses) == 1:
            tag, dmrs = self.clause_dmrs(proposition.clauses[0])
            assert tag in ('modifier', 'noun', 'quantifier')
            if tag == 'modifier':
                dmrs = DmrsRealizer.empty_predicate.compose(dmrs, {'noun': 'noun'})
                dmrs = DmrsRealizer.empty_clause.compose(dmrs, {'noun': 'noun'})
                dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'noun': 'rstr'})
            elif tag == 'noun':
                dmrs = DmrsRealizer.empty_clause.compose(dmrs, {'noun': 'noun'})
                dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'noun': 'rstr'})
        else:
            dmrs = DmrsRealizer.propositions[proposition.connective]
            # better with first, between, last?
            dmrs = dmrs.compose(self.clause_dmrs(proposition.clauses[-1]), {'arg2': 'rel'})
            for clause in reversed(proposition.clauses[1:-1]):
                dmrs = dmrs.compose(self.clause_dmrs(clause), {'arg1': 'rel'})
                dmrs = DmrsRealizer.propositions[proposition.connective].compose(dmrs, {'arg2': 'con'})
            dmrs = dmrs.compose(self.clause_dmrs(proposition.clauses[0]), {'arg1': 'rel'})
        return dmrs


realizer = DmrsRealizer


DmrsRealizer.empty_predicate = ComposableDmrs.parse('[noun]:_shape_n_1 x[???+?]')
DmrsRealizer.empty_relation = ComposableDmrs.parse('[noun]:pred x? <-1- ***[rel]:_be_v_id e[ppi--] -2-> [ref]:node')
DmrsRealizer.adjective_relation = ComposableDmrs.parse('[noun]:pred x? <-1- ***[rel]:pred e[ppi--]')
DmrsRealizer.empty_singular_quantifier = ComposableDmrs.parse('[quant]:_a_q --> [rstr]:pred x[?s???]')
DmrsRealizer.empty_plural_quantifier = ComposableDmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???]')
DmrsRealizer.empty_clause = ComposableDmrs.parse('[noun]:node <-1- ***[rel]:_be_v_there e[ppi--]')


DmrsRealizer.modifiers = {
    'shape': {
        'square': ComposableDmrs.parse('[noun]:_square_n_1 x[???+?]'),
        'rectangle': ComposableDmrs.parse('[noun]:_rectangle_n_1 x[???+?]'),
        'triangle': ComposableDmrs.parse('[noun]:_triangle_n_1 x[???+?]'),
        'pentagon': ComposableDmrs.parse('[noun]:_pentagon_n_1 x[???+?]'),
        'cross': ComposableDmrs.parse('[noun]:_cross_n_1 x[???+?]'),
        'circle': ComposableDmrs.parse('[noun]:_circle_n_1 x[???+?]'),
        'semicircle': ComposableDmrs.parse('[noun]:_semicircle_n_1 x[???+?]'),
        'ellipse': ComposableDmrs.parse('[noun]:_ellipse_n_1 x[???+?]')
    },

    'color': {
        'black': ComposableDmrs.parse('[mod]:_black_a_1 e? =1=> [noun]:node'),
        'red': ComposableDmrs.parse('[mod]:_red_a_1 e? =1=> [noun]:node'),
        'green': ComposableDmrs.parse('[mod]:_green_a_1 e? =1=> [noun]:node'),
        'blue': ComposableDmrs.parse('[mod]:_blue_a_1 e? =1=> [noun]:node'),
        'yellow': ComposableDmrs.parse('[mod]:_yellow_a_1 e? =1=> [noun]:node'),
        'magenta': ComposableDmrs.parse('[mod]:_magenta_a_1 e? =1=> [noun]:node'),
        'cyan': ComposableDmrs.parse('[mod]:_cyan_a_1 e? =1=> [noun]:node'),
        'white': ComposableDmrs.parse('[mod]:_white_a_1 e? =1=> [noun]:node')
    },

    'texture': {
        'solid': ComposableDmrs.parse('[noun]:node')
    }
}

DmrsRealizer.modifier_is_noun = {
    'shape': {'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'},
    'color': set(),
    'texture': set()
}


DmrsRealizer.relations = {
    'left': ComposableDmrs.parse('[noun]:pred x? <-1- ***[rel]:_to_p e[ppi--] -2-> _left_n_of x[_s___] <-- _the_q; :_left_n_of -1-> [ref]:pred x[?s???]'),
    'right': ComposableDmrs.parse('[noun]:pred x? <-1- ***[rel]:_to_p e[ppi--] -2-> _right_n_of x[_s___] <-- _the_q; :_right_n_of -1-> [ref]:pred x[?s???]'),
    'above': ComposableDmrs.parse('[noun]:pred x? <-1- ***[rel]:_above_p e[ppi--] -2-> [ref]:pred x[?s???]'),
    'below': ComposableDmrs.parse('[noun]:pred x? <-1- ***[rel]:_below_p e[ppi--] -2-> [ref]:pred x[?s???]')
}


DmrsRealizer.quantifiers = {
    'absolute': {
        'eq': {
            1: ComposableDmrs.parse('[quant]:_the_q --> [rstr]:pred x[?s???]'),  # False, True, 1, False
        },
        'geq': {
            1: ComposableDmrs.parse('[quant]:_a_q --> [rstr]:pred x[?s???]'),  # False, False, 1, False
            # ComposableDmrs.parse('[quant]:_some_q --> [rstr]:pred x[?p???]')  # False, False, 1, True
            2: ComposableDmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???] <=1= card(2) e')  # False
        }
    },

    'relative': {
        'eq': {
            0.0: ComposableDmrs.parse('[quant]:_no_q --> [rstr]:pred x[?s???]'),  # False, True, 0, False
            1.0: ComposableDmrs.parse('[quant]:_all_q --> [rstr]:pred x[?p???]')  # True, False, 1.0, True
            # ComposableDmrs.parse('[quant]:_every_q --> [rstr]:pred x[?s???]')  # True, False, 1.0, False
        },
        'geq': {
            0.6: ComposableDmrs.parse('[quant]:_most_q --> [rstr]:pred x[?p???]')  # True, True, 0.5, True
        }
    }

    # 'mixture'/'composition': ...
}

DmrsRealizer.quantifier_by_name = {
    'no': ('relative', 'eq', 0.0),
    'the': ('absolute', 'eq', 1),
    'a': ('absolute', 'geq', 1),
    'two': ('absolute', 'geq', 2),
    'most': ('relative', 'geq', 0.6),
    'all': ('relative', 'eq', 1.0)
}

DmrsRealizer.quantifier_requires_plural = {
    ('absolute', 'geq', 2), ('relative', 'eq', 1.0), ('relative', 'geq', 0.6)
}


DmrsRealizer.captions = {
    'and': ComposableDmrs.parse('[arg1]:node <-l- [con]:_and_c e[ppi--] -r-> [arg2]:node'),
    'or': ComposableDmrs.parse('[arg1]:node <-l- [con]:_or_c e[ppi--] -r-> [arg2]:node')
}
