import os
import re
import subprocess
from shapeworld import CaptionRealizer
from shapeworld.caption import Predicate, Modifier, Noun, Relation, Quantifier, Proposition
from shapeworld.realizers.dmrs.dmrs import Dmrs

from datetime import datetime
import sys


class DmrsRealizer(CaptionRealizer):

    def __init__(self):
        super(DmrsRealizer, self).__init__()
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')
        self.ace_path = os.path.join(directory, 'ace')
        self.erg_path = os.path.join(directory, 'erg-shapeworld.dat')
        self.regex = re.compile(pattern=r'(NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\])|(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)')

    def realize(self, captions):
        assert all(isinstance(caption, Proposition) for caption in captions)
        try:
            ace = subprocess.Popen([self.ace_path, '-g', self.erg_path, '-1Te'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(datetime.now().strftime('%H:%M:%S'))
            print(e.strerror)
            print(sys.exc_info()[0])
            raise
        mrs = '\n'.join(self.proposition_dmrs(caption).get_mrs() for caption in captions) + '\n'
        stdout_data, stderr_data = ace.communicate(mrs.encode())
        assert all(self.regex.match(line) for line in stderr_data.decode('utf-8').splitlines()), '\n\n' + '\n'.join('{}\n{}\n{}\n'.format(line, '', self.proposition_dmrs(caption).get_mrs()) for line, caption in zip(stderr_data.decode('utf-8').splitlines(), captions) if not self.regex.match(line))  # self.proposition_dmrs(caption).dumps_xml()
        caption_strings = [line for line in stdout_data.decode('utf-8').splitlines() if line]
        assert len(caption_strings) == len(captions)
        for n, caption in enumerate(caption_strings):
            caption = caption.lower()
            caption = caption.replace('.', ' .').replace(',', ' ,')
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
        if modifier.modtype == 'relation':
            _, dmrs = self.relation_dmrs(relation=modifier.value)
            DmrsRealizer.relation_modifier.compose(dmrs, {'noun': 'noun', 'mod': 'rel'})
            return 'modifier', dmrs
        else:
            assert modifier.modtype in DmrsRealizer.modifiers and modifier.value in DmrsRealizer.modifiers[modifier.modtype]
            dmrs = DmrsRealizer.modifiers[modifier.modtype][modifier.value]
            tag = 'noun' if DmrsRealizer.modifier_is_noun[modifier.modtype] else 'modifier'
            return tag, dmrs

    def relation_dmrs(self, relation):
        assert relation.reltype in DmrsRealizer.relations and relation.value in DmrsRealizer.relations[relation.reltype]
        dmrs = DmrsRealizer.relations[relation.reltype][relation.value]
        # reference
        tag, ref_dmrs = self.predicate_dmrs(relation.reference)
        dmrs = dmrs.compose(ref_dmrs, {'ref': 'noun'})
        if tag == 'relation':
            dmrs = DmrsRealizer.relation_modifier.compose(dmrs, {'noun': 'noun', 'mod': 'rel'})
            tag = 'modifier'
        if tag == 'modifier':
            dmrs = dmrs.compose(DmrsRealizer.empty_predicate, {'ref': 'noun'})
        dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'ref': 'rstr'})
        # comparison
        if relation.comparison is not None:
            tag, comp_dmrs = self.predicate_dmrs(relation.comparison)
            dmrs = dmrs.compose(comp_dmrs, {'comp': 'noun'})
            if tag == 'relation':
                dmrs = DmrsRealizer.relation_modifier.compose(dmrs, {'noun': 'noun', 'mod': 'rel'})
                tag = 'modifier'
            if tag == 'modifier':
                dmrs = dmrs.compose(DmrsRealizer.empty_predicate, {'comp': 'noun'})
            dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'comp': 'rstr'})
        return 'relation', dmrs

    def noun_dmrs(self, noun):
        if not noun.predicates:
            return 'noun', DmrsRealizer.empty_predicate
        dmrs = None
        contains_noun = False
        for predicate in noun.predicates:
            tag, pred_dmrs = self.predicate_dmrs(predicate)
            if tag == 'noun':
                assert not contains_noun
                contains_noun = True
            elif tag == 'relation':
                pred_dmrs = DmrsRealizer.relation_modifier.compose(pred_dmrs, {'noun': 'noun', 'mod': 'rel'})
            if dmrs is None:
                dmrs = pred_dmrs
            else:
                dmrs = dmrs.compose(pred_dmrs, {'noun': 'noun'})
        tag = 'noun' if contains_noun else 'modifier'
        return tag, dmrs

    def quantifier_dmrs(self, quantifier):
        # tag, dmrs = self.predicate_dmrs(quantifier.body)
        # if tag == 'modifier':
        #     dmrs = DmrsRealizer.adjective_relation.compose(dmrs, {'rel': 'mod'})
        # elif tag == 'noun':
        #     dmrs = DmrsRealizer.empty_relation.compose(dmrs, {'ref': 'noun'})
        #     require_plural = (quantifier.qtype, quantifier.qrange, quantifier.quantity) in DmrsRealizer.quantifier_requires_plural
        #     if require_plural:
        #         dmrs = dmrs.compose(DmrsRealizer.empty_plural_quantifier, {'ref': 'rstr'})
        #     else:
        #         dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'ref': 'rstr'})
        # tag, rstr_dmrs = self.predicate_dmrs(quantifier.restrictor)
        # assert tag in ('modifier', 'noun')
        # dmrs = dmrs.compose(rstr_dmrs, {'noun': 'noun'})
        # if tag == 'modifier':
        #     dmrs = dmrs.compose(DmrsRealizer.empty_predicate, {'noun': 'noun'})
        # dmrs = dmrs.compose(DmrsRealizer.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity], {'noun': 'rstr'})
        tag, body_dmrs = self.predicate_dmrs(quantifier.body)
        if tag == 'modifier':
            body_dmrs = DmrsRealizer.modifier_relation.compose(body_dmrs, {'rel': 'mod'})
        elif tag == 'noun':
            body_dmrs = DmrsRealizer.empty_relation.compose(body_dmrs, {'ref': 'noun'})
            require_singular = (quantifier.qtype, quantifier.qrange, quantifier.quantity) in DmrsRealizer.quantifier_requires_singular
            if require_singular:
                body_dmrs = body_dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'ref': 'rstr'})
            else:
                body_dmrs = body_dmrs.compose(DmrsRealizer.empty_plural_quantifier, {'ref': 'rstr'})
        tag, rstr_dmrs = self.predicate_dmrs(quantifier.restrictor)
        if tag == 'relation':
            rstr_dmrs = DmrsRealizer.relation_modifier.compose(rstr_dmrs, {'noun': 'noun', 'mod': 'rel'})
            tag = 'modifier'
        if tag == 'modifier':
            rstr_dmrs = rstr_dmrs.compose(DmrsRealizer.empty_predicate, {'noun': 'noun'})
        dmrs = DmrsRealizer.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity]
        dmrs = dmrs.compose(rstr_dmrs, {'rstr': 'noun'})
        dmrs = dmrs.compose(body_dmrs, {'noun': 'noun'})
        return 'quantifier', dmrs

    def proposition_dmrs(self, proposition):
        assert len(proposition.clauses) >= 1
        clauses_dmrs = []
        for clause in proposition.clauses:
            tag, dmrs = self.clause_dmrs(clause)
            if tag == 'relation':
                dmrs = DmrsRealizer.relation_modifier.compose(dmrs, {'noun': 'noun', 'mod': 'rel'})
                tag = 'modifier'
            if tag == 'modifier':
                dmrs = DmrsRealizer.empty_predicate.compose(dmrs, {'noun': 'noun'})
                tag = 'noun'
            if tag == 'noun':
                dmrs = DmrsRealizer.empty_clause.compose(dmrs, {'noun': 'noun'})
                dmrs = dmrs.compose(DmrsRealizer.empty_singular_quantifier, {'noun': 'rstr'})
            clauses_dmrs.append(dmrs)
        # connective = None?
        if len(clauses_dmrs) == 1:
            dmrs = clauses_dmrs[0]
        else:
            dmrs = DmrsRealizer.propositions[proposition.connective]
            if isinstance(dmrs, Dmrs):
                connective_dmrs = intermediate_dmrs = dmrs
                first_dmrs = None
            else:
                connective_dmrs, intermediate_dmrs, first_dmrs = dmrs
            dmrs = connective_dmrs.compose(clauses_dmrs[-1], {'arg2': 'rel'})
            for clause_dmrs in reversed(clauses_dmrs[1:-1]):
                dmrs = dmrs.compose(clause_dmrs, {'arg1': 'rel'})
                dmrs = intermediate_dmrs.compose(dmrs, {'arg2': 'con'})
            dmrs = dmrs.compose(clauses_dmrs[0], {'arg1': 'rel'})
            if first_dmrs:
                dmrs = dmrs.compose(first_dmrs, {'arg1': 'arg'})
        return dmrs


realizer = DmrsRealizer


DmrsRealizer.empty_predicate = Dmrs.parse('[noun]:_shape_n_1 x[3??+?]')
DmrsRealizer.empty_relation = Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_be_v_id e[ppi--] -2-> [ref]:node')
DmrsRealizer.modifier_relation = Dmrs.parse('[noun]:pred x? <-1- ***[rel]:pred e[ppi--]')
DmrsRealizer.relation_modifier = Dmrs.parse('[noun]:pred x? <=1= [mod]:pred e[ppi--]')
DmrsRealizer.empty_singular_quantifier = Dmrs.parse('[quant]:_a_q --> [rstr]:pred x[?s???]')
DmrsRealizer.empty_plural_quantifier = Dmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???]')
DmrsRealizer.empty_clause = Dmrs.parse('[noun]:node <-1- ***[rel]:_be_v_there e[ppi--]')


DmrsRealizer.modifiers = {
    'shape': {
        'square': Dmrs.parse('[noun]:_square_n_1 x[3??+?]'),
        'rectangle': Dmrs.parse('[noun]:_rectangle_n_1 x[3??+?]'),
        'triangle': Dmrs.parse('[noun]:_triangle_n_1 x[3??+?]'),
        'pentagon': Dmrs.parse('[noun]:_pentagon_n_1 x[3??+?]'),
        'cross': Dmrs.parse('[noun]:_cross_n_1 x[3??+?]'),
        'circle': Dmrs.parse('[noun]:_circle_n_1 x[3??+?]'),
        'semicircle': Dmrs.parse('[noun]:_semicircle_n_1 x[3??+?]'),
        'ellipse': Dmrs.parse('[noun]:_ellipse_n_1 x[3??+?]')
    },

    'color': {
        'black': Dmrs.parse('[mod]:_black_a_1 e? =1=> [noun]:node'),
        'red': Dmrs.parse('[mod]:_red_a_1 e? =1=> [noun]:node'),
        'green': Dmrs.parse('[mod]:_green_a_1 e? =1=> [noun]:node'),
        'blue': Dmrs.parse('[mod]:_blue_a_1 e? =1=> [noun]:node'),
        'yellow': Dmrs.parse('[mod]:_yellow_a_1 e? =1=> [noun]:node'),
        'magenta': Dmrs.parse('[mod]:_magenta_a_1 e? =1=> [noun]:node'),
        'cyan': Dmrs.parse('[mod]:_cyan_a_1 e? =1=> [noun]:node'),
        'white': Dmrs.parse('[mod]:_white_a_1 e? =1=> [noun]:node')
    },

    'texture': {
        'solid': Dmrs.parse('[noun]:node')
    },

    'x-max': {
        -1: Dmrs.parse('[mod]:_leftmost_a_1 e? =1=> [noun]:node'),
        1: Dmrs.parse('[mod]:_rightmost_a_1 e? =1=> [noun]:node')
    },

    'y-max': {
        -1: Dmrs.parse('[mod]:_topmost_a_1 e? =1=> [noun]:node'),
        1: Dmrs.parse('superl e =1=> [mod]:_low_a_1 e? =1=> [noun]:node')
    },

    'size-max': {
        -1: Dmrs.parse('superl e =1=> [mod]:_small_a_1 e? =1=> [noun]:node'),
        1: Dmrs.parse('superl e =1=> [mod]:_big_a_1 e? =1=> [noun]:node')
    },

    'shade-max': {
        -1: Dmrs.parse('superl e =1=> [mod]:_dark_a_1 e? =1=> [noun]:node'),
        1: Dmrs.parse('superl e =1=> [mod]:_light_a_1 e? =1=> [noun]:node')
    }
}

DmrsRealizer.modifier_by_name = {
    'square': ('shape', 'square'),
    'rectangle': ('shape', 'rectangle'),
    'triangle': ('shape', 'triangle'),
    'pentagon': ('shape', 'pentagon'),
    'cross': ('shape', 'cross'),
    'circle': ('shape', 'circle'),
    'semicircle': ('shape', 'semicircle'),
    'ellipse': ('shape', 'ellipse'),

    'black': ('color', 'black'),
    'red': ('color', 'red'),
    'green': ('color', 'green'),
    'blue': ('color', 'blue'),
    'yellow': ('color', 'yellow'),
    'magenta': ('color', 'magenta'),
    'cyan': ('color', 'cyan'),
    'white': ('color', 'white'),

    'solid': ('texture', 'solid'),

    'leftmost': ('x-max', -1),
    'rightmost': ('x-max', 1),

    'topmost': ('y-max', -1),
    'lowest': ('y-max', 1),

    'smallest': ('size-max', -1),
    'biggest': ('size-max', 1),

    'darkest': ('shade-max', -1),
    'lightest': ('shade-max', 1)
}

DmrsRealizer.modifier_is_noun = {
    'shape': {'square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'},
    'color': set(),
    'texture': set(),
    'x-max': set(),
    'y-max': set(),
    'size-max': set(),
    'shade-max': set()
}


# singular in reference required ???????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????
DmrsRealizer.relations = {
    'x-rel': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_to_p e[ppi--] -2-> _left_n_of x[_s___] <-- _the_q; :_left_n_of -1-> [ref]:pred x[?s???]'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_to_p e[ppi--] -2-> _right_n_of x[_s___] <-- _the_q; :_right_n_of -1-> [ref]:pred x[?s???]')
    },

    'y-rel': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_above_p e[ppi--] -2-> [ref]:pred x?'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_below_p e[ppi--] -2-> [ref]:pred x?')
    },

    'z-rel': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_behind_p e[ppi--] -2-> [ref]:pred x?'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_in+front+of_p e[ppi--] -2-> [ref]:pred x?'),
    },

    'proximity-max': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_close_a_to e[ppi--] -2-> [ref]:pred x?; superl e =1=> :rel'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_far_a_from e[ppi--] -2-> [ref]:pred x?; superl e =1=> :rel')
    },

    'proximity-rel': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_close_a_to e[ppi--] -2-> [ref]:pred x?; :rel <=1= more_comp e -2-> [comp]:pred x?'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_far_a_from e[ppi--] -2-> [ref]:pred x?; :rel <=1= more_comp e -2-> [comp]:pred x?'),
    },

    'size-rel': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_small_a_1 e[ppi--] <=1= comp e -2-> [ref]:pred x?'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_big_a_1 e[ppi--] <=1= comp e -2-> [ref]:pred x?')
    },

    'shade-rel': {
        -1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_dark_a_1 e[ppi--] <=1= comp e -2-> [ref]:pred x?'),
        1: Dmrs.parse('[noun]:pred x? <-1- ***[rel]:_light_a_1 e[ppi--] <=1= comp e -2-> [ref]:pred x?')
    }
}

DmrsRealizer.relation_by_name = {
    'left': ('x-rel', -1),
    'right': ('x-rel', 1),

    'above': ('y-rel', -1),
    'below': ('y-rel', 1),

    'behind': ('z-rel', -1),
    'in_front_of': ('z-rel', 1),

    'closest': ('proximity-max', -1),
    'farthest': ('proximity-max', 1),

    'closer': ('proximity-rel', -1),
    'farther': ('proximity-rel', 1),

    'smaller': ('size-rel', -1),
    'bigger': ('size-rel', 1),

    'darker': ('shade-rel', -1),
    'lighter': ('shade-rel', 1)
}


DmrsRealizer.quantifiers = {
    'absolute': {
        'eq': {
            1: Dmrs.parse('[quant]:_the_q --> [rstr]:pred x[?s???]'),
            2: Dmrs.parse('[quant]:_both_q --> [rstr]:pred x[?p???] <=1= card(2) e'),
            3: Dmrs.parse('[quant]:_the_q --> [rstr]:pred x[?p???] <=1= card(3) e'),
            4: Dmrs.parse('[quant]:_the_q --> [rstr]:pred x[?p???] <=1= card(4) e'),
            5: Dmrs.parse('[quant]:_the_q --> [rstr]:pred x[?p???] <=1= card(5) e')
        },
        'geq': {
            1: Dmrs.parse('[quant]:_a_q --> [rstr]:pred x[?s???]'),  # _some_q
            2: Dmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???] <=1= card(2) e'),  # many
            3: Dmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???] <=1= card(3) e'),
            4: Dmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???] <=1= card(4) e'),
            5: Dmrs.parse('[quant]:udef_q --> [rstr]:pred x[?p???] <=1= card(5) e')
        }
    },

    'relative': {
        'eq': {
            0.0: Dmrs.parse('[quant]:_no_q --> [rstr]:pred x[?s???]'),
            0.5: Dmrs.parse('[quant]:_half_q --> [noun]:part_of x[3p???] -1-> [rstr]:pred x[?p???] <-- _the_q'),
            1.0: Dmrs.parse('[quant]:_all_q --> [rstr]:pred x[?p???]'),  # _every_q, _the_q (+ pl), any, each
        },
        'geq': {
            0.6: Dmrs.parse('[quant]:_most_q --> [rstr]:pred x[?p???]')
        }
    }

    # 'mixture'/'composition': ...
}

DmrsRealizer.quantifier_by_name = {
    'the': ('absolute', 'eq', 1),
    'both': ('absolute', 'eq', 2),
    'the_three': ('absolute', 'eq', 3),
    'the_four': ('absolute', 'eq', 4),
    'the_five': ('absolute', 'eq', 5),

    'a': ('absolute', 'geq', 1),
    'two': ('absolute', 'geq', 2),
    'three': ('absolute', 'geq', 3),
    'four': ('absolute', 'geq', 4),
    'five': ('absolute', 'geq', 5),

    'no': ('relative', 'eq', 0.0),
    'half': ('relative', 'eq', 0.5),
    'all': ('relative', 'eq', 1.0),

    'most': ('relative', 'geq', 0.6)
}

DmrsRealizer.quantifier_requires_singular = {
    ('absolute', 'eq', 1), ('absolute', 'geq', 1), ('relative', 'eq', 0.0)
}


DmrsRealizer.propositions = {
    'conjunction': Dmrs.parse('[arg1]:node <-l- ***[con]:_and_c e[ppi--] -r-> [arg2]:node'),
    'disjunction': Dmrs.parse('[arg1]:node <-l- ***[con]:_or_c e[ppi--] -r-> [arg2]:node'),
    'exclusive-disjunction': (Dmrs.parse('[arg1]:node <-l- ***[con]:_or_c e[ppi--] -r-> [arg2]:node'), Dmrs.parse('[arg1]:node <-l- ***[con]:_or_c e[ppi--] -r-> [arg2]:node'), Dmrs.parse('_either_a_also i =1=> [arg]:node'))  # Dmrs.parse('[arg1]:node <-l- ***[con]:implicit_conj e[ppi--] -r-> [arg2]:node')
}
