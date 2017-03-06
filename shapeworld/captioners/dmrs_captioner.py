from copy import deepcopy
import os
from random import choice
import re
import subprocess

from pydmrs.components import Pred
from pydmrs.core import Link, ListDmrs
from pydmrs.graphlang.graphlang import parse_graphlang

from shapeworld.caption import Caption
from shapeworld.captioner import WorldCaptioner


resources_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources')


class AnchorDmrs(ListDmrs):
    __slots__ = ('nodes', 'links', 'index', 'top', 'anchors')

    def __init__(self, nodes=(), links=(), index=None, top=None):
        super().__init__(nodes=nodes, links=links, index=index, top=top)
        self.anchors = None

    def compose(self, other, fusion):
        assert isinstance(other, AnchorDmrs)
        composition = deepcopy(self)
        nodeid_mapping = dict()
        for anchor1, anchor2 in fusion.items():
            node1 = composition.anchors[anchor1]
            node2 = other.anchors[anchor2]
            node1.unify(node2)
            nodeid_mapping[node2.nodeid] = node1.nodeid
        for nodeid2 in other:
            if nodeid2 in nodeid_mapping:
                continue
            node1 = deepcopy(other[nodeid2])
            node1.nodeid = None
            nodeid_mapping[nodeid2] = composition.add_node(node1)
        for link2 in other.iter_links():
            link1 = Link(nodeid_mapping[link2.start], nodeid_mapping[link2.end], link2.rargname, link2.post)
            composition.add_link(link1)
        if composition.index is None and other.index is not None:
            composition.index = composition[nodeid_mapping[other.index.nodeid]]
        if composition.top is None and other.top is not None:
            composition.top = composition[nodeid_mapping[other.top.nodeid]]
        return composition

    def get_mrs(self):
        labels = dict(zip(self, range(1, len(self) + 1)))
        redirected = []
        quantifiers = dict()
        for link in self.iter_links():
            assert isinstance(link.start, int) and isinstance(link.end, int)
            assert isinstance(link.rargname, str) or (link.rargname is None and link.post == 'EQ')  # ('ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG', 'RSTR', 'BODY', 'L-INDEX', 'R-INDEX', 'L-HNDL', 'R-HNDL')
            assert link.post in ('NEQ', 'EQ', 'H', 'HEQ')
            if link.post == 'EQ':
                upper, lower = (link.start, link.end) if link.start > link.end else (link.end, link.start)
                while lower in redirected:
                    lower = labels[lower]
                labels[upper] = labels[lower]
                redirected.append(upper)
            elif link.rargname == 'RSTR' and link.post == 'H':
                quantifiers[link.start] = link.end

        predicates = dict()
        cargs = dict()
        variables = dict()
        index = max(labels.values())
        for node in self.iter_nodes():
            assert node.nodeid in self
            assert node.pred is not None
            predicates[node.nodeid] = str(node.pred)
            if node.carg is not None:
                cargs[node.nodeid] = node.carg
            if node.nodeid not in quantifiers:
                assert node.sortinfo is not None
                index += 1
                variables[node.nodeid] = (node.sortinfo.cvarsort + str(index), node.sortinfo)

        args = dict()
        hcons = {0: self.top.nodeid}
        for link in self.iter_links():
            if link.start not in args and link.rargname is not None:
                args[link.start] = dict()
            if link.post == 'NEQ':
                assert link.rargname not in args[link.start]
                args[link.start][link.rargname] = variables[link.end][0]
            elif link.post == 'EQ':
                if link.rargname is not None:
                    assert link.rargname not in args[link.start]
                    args[link.start][link.rargname] = variables[link.end][0]
            elif link.post == 'H':
                assert link.rargname not in args[link.start]
                index += 1
                args[link.start][link.rargname] = 'h' + str(index)
                hcons[index] = link.end
            elif link.post == 'HEQ':
                args[link.start][link.rargname] = 'h' + str(labels[link.end])
            else:
                assert False

        elempreds = []
        for nodeid in self:
            carg_string = 'CARG: "{}" '.format(cargs[nodeid]) if nodeid in cargs else ''
            if nodeid in quantifiers:
                intrinsic_string = variables[quantifiers[nodeid]][0]
            else:
                intrinsic_string = '{} [ {} {}]'.format(variables[nodeid][0], variables[nodeid][1].cvarsort, ''.join('{}: {} '.format(feature.upper(), value.lower()) for feature, value in variables[nodeid][1].iter_specified()))
            args_string = ''.join('{}: {} '.format(role.upper(), arg) for role, arg in args[nodeid].items()) if nodeid in args else ''
            elempred_string = '[ {}_rel LBL: h{} {}ARG0: {} {}]'.format(predicates[nodeid], labels[nodeid], carg_string, intrinsic_string, args_string)
            elempreds.append(elempred_string)

        top_string = '' if self.top is None else 'TOP: h0 '
        index_string = '' if self.index is None else 'INDEX: {} '.format(variables[self.index.nodeid][0])
        eps_string = '  '.join(elempreds)
        hcons_string = ' '.join('h{} qeq h{}'.format(*qeq) for qeq in hcons.items())
        mrs_string = '[ {}{}RELS: < {} > HCONS: < {} > ]'.format(top_string, index_string, eps_string, hcons_string)
        return mrs_string

    @staticmethod
    def parse(string):
        dmrs_variants = []
        for dmrs in string.split(';;;'):
            anchors = dict()
            dmrs = parse_graphlang(dmrs, cls=AnchorDmrs, anchors=anchors)
            dmrs.anchors = anchors
            dmrs_variants.append(dmrs)
        return dmrs_variants


class DmrsCaption(Caption):
    __slots__ = ('category', 'agreeing_entities', 'agreement', 'dmrs')

    def __init__(self, category, agreeing_entities, agreement, dmrs):
        super().__init__()
        self.category = category
        self.agreeing_entities = agreeing_entities
        self.agreement = agreement
        self.dmrs = dmrs

    def get_mrs(self):
        dmrs = self.dmrs
        for node in dmrs.iter_nodes():
            if type(node.pred) is Pred:
                dmrs.remove_node(node.nodeid)
                dmrs.remove_links(link for link in dmrs.iter_links() if link.start == node.nodeid or link.end == node.nodeid)
            if node.sortinfo is None:
                continue
            node.sortinfo = node.sortinfo.__class__(**{key: None if node.sortinfo[key] in ('u', '?') else node.sortinfo[key] for key in node.sortinfo if key != 'cvarsort'})
        return dmrs.get_mrs()


class DmrsCaptionerComponent(object):

    def instantiate(self):
        raise NotImplementedError


class DmrsCaptioner(WorldCaptioner, DmrsCaptionerComponent):

    def __init__(self, caption_size, words):
        super().__init__(caption_size, words)
        self.regex = re.compile(pattern=r'(NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\])|(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)')

    def realize(self, captions):
        ace_path = os.path.join(resources_directory, 'ace')
        erg_path = os.path.join(resources_directory, 'erg-shapeworld.dat')
        ace = subprocess.Popen([ace_path, '-g', erg_path, '-1Te'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mrs = '\n'.join(caption.get_mrs() for caption in captions) + '\n'
        stdout_data, stderr_data = ace.communicate(mrs.encode())
        assert all(self.regex.match(string=line) for line in stderr_data.decode('utf-8').splitlines())
        caption_strings = stdout_data.decode('utf-8').splitlines()
        assert len(caption_strings) == len(captions)
        for n, caption in enumerate(caption_strings):
            caption = caption.lower()
            caption = caption.replace('.', ' .')
            captions[n] = caption.split()
        return captions


class Quantifier(DmrsCaptionerComponent):

    def __init__(self, relative, strict, threshold, plural, dmrs_variants):
        self.relative = relative
        self.strict = strict
        self.threshold = threshold
        self.plural = plural
        self.dmrs_variants = dmrs_variants

    def instantiate(self, rstr, body):
        if body is None:  # ???
            assert rstr.category == 'noun'
            category = 'noun'

            def agreeing_entities(entities):
                return rstr.agreeing_entities(entities)

            dmrs = deepcopy(choice(self.dmrs_variants))
            dmrs = dmrs.compose(rstr.dmrs, {'noun': 'noun'})
            return DmrsCaption(category, agreeing_entities, DmrsCaption.none_agreement, dmrs)

        else:
            assert rstr.category == 'noun' and body.category in ('noun', 'modifier', 'relation')  # ???
            category = 'quantifier'

            def agreeing_entities(entities):
                return None

            threshold = self.threshold
            if self.relative:  # agreement value between 0 and 1???
                if self.strict:  # all_count is not zero !!!
                    def count_agreement(all_count, agreeing_count):
                        return float(all_count > 0 and agreeing_count / all_count > threshold)
                else:
                    def count_agreement(all_count, agreeing_count):
                        return float(all_count > 0 and agreeing_count / all_count >= threshold)
            else:
                if self.strict:
                    def count_agreement(all_count, agreeing_count):
                        return float(agreeing_count == threshold)
                else:
                    def count_agreement(all_count, agreeing_count):
                        return float(agreeing_count >= threshold)

            def agreement(world):
                rstr_entities = rstr.agreeing_entities(world.entities)
                rstr_count = len(rstr_entities)
                body_count = len(body.agreeing_entities(rstr_entities))
                return count_agreement(rstr_count, body_count)

            dmrs = deepcopy(choice(self.dmrs_variants))
            dmrs = dmrs.compose(rstr.dmrs, {'noun': 'noun'})
            if isinstance(body.dmrs, dict):
                dmrs = dmrs.compose(body.dmrs[self.plural], {'noun': 'noun'})
            else:
                dmrs = dmrs.compose(body.dmrs, {'noun': 'noun'})
            return DmrsCaption(category, agreeing_entities, agreement, dmrs)


class Noun(DmrsCaptionerComponent):

    def __init__(self, property, value, dmrs_variants):
        self.noun_modifier = Modifier(property, value, (None, dict()))
        self.dmrs_variants = dmrs_variants

    def instantiate(self, modifiers):
        assert all(modifier.category == 'modifier' for modifier in modifiers)
        noun_caption = self.noun_modifier.instantiate()
        category = 'noun'

        def agreeing_entities(entities):
            entities = noun_caption.agreeing_entities(entities)
            for modifier in modifiers:
                entities = modifier.agreeing_entities(entities)
            return entities

        dmrs = deepcopy(choice(self.dmrs_variants))
        for modifier in modifiers:
            dmrs = dmrs.compose(modifier.dmrs, {'noun': 'noun'})
        return DmrsCaption(category, agreeing_entities, DmrsCaption.none_agreement, dmrs)


class Modifier(DmrsCaptionerComponent):

    def __init__(self, property, value, dmrs_variants):
        self.property = property
        self.value = value
        self.dmrs_variants = dmrs_variants

    def instantiate(self):
        category = 'modifier'
        # with agreement values?

        value = self.value
        if self.property == 'shape':
            def agreeing_entities(entities):
                return [entity for entity in entities if entity.shape == value]
        elif self.property == 'shape-set':
            def agreeing_entities(entities):
                return [entity for entity in entities if entity.shape in value]
        elif self.property == 'color':
            def agreeing_entities(entities):
                return [entity for entity in entities if str(entity.color) == value]
        elif self.property == 'shade-max':
            def agreeing_entities(entities):
                # all same color?
                max_entity = None
                max_shade = -1.0
                for entity in entities:
                    if entity.color.shade * value > max_shade:
                        max_shade = entity.color.shade * value
                        max_entity = entity
                if max_entity is None:
                    return []
                else:
                    return [max_entity]
        elif self.property == 'location-max':
            def agreeing_entities(entities):
                max_entity = None
                max_location = -1.0
                for entity in entities:
                    if sum(entity.location * value) > max_location:
                        max_location = entity.color.shade * value
                        max_entity = entity
                if max_entity is None:
                    return []
                else:
                    return [max_entity]
        else:
            assert False

        dmrs = deepcopy(choice(self.dmrs_variants))
        return DmrsCaption(category, agreeing_entities, DmrsCaption.none_agreement, dmrs)


DmrsCaptionerComponent.quantifiers = {
    'no': Quantifier(False, True, 0, False, AnchorDmrs.parse('[quant]:_no_q --> [noun]:pred x[?s???]')),
    'the': Quantifier(False, True, 1, False, AnchorDmrs.parse('[quant]:_the_q --> [noun]:pred x[?s???]')),
    'a': Quantifier(False, False, 1, False, AnchorDmrs.parse('[quant]:_a_q --> [noun]:pred x[?s???]')),
    'some': Quantifier(False, False, 1, True, AnchorDmrs.parse('[quant]:_some_q --> [noun]:pred x[?p???]')),
    'two': Quantifier(False, False, 2, True, AnchorDmrs.parse('[quant]:udef_q --> [noun]:pred x[?p???] <=1= card(2) e')),
    'most': Quantifier(True, True, 0.5, True, AnchorDmrs.parse('[quant]:_most_q --> [noun]:pred x[?p???]')),
    'all': Quantifier(True, False, 1.0, True, AnchorDmrs.parse('[quant]:_all_q --> [noun]:pred x[?p???]')),
    'every': Quantifier(True, False, 1.0, False, AnchorDmrs.parse('[quant]:_every_q --> [noun]:pred x[?s???]'))}


DmrsCaptionerComponent.nouns = {
    'square': Noun('shape', 'square', AnchorDmrs.parse('[noun]:_square_n_1 x[???+?]')),
    'rectangle': Noun('shape', 'rectangle', AnchorDmrs.parse('[noun]:_rectangle_n_1 x[???+?]')),
    'triangle': Noun('shape', 'triangle', AnchorDmrs.parse('[noun]:_triangle_n_1 x[???+?]')),
    'pentagon': Noun('shape', 'pentagon', AnchorDmrs.parse('[noun]:_pentagon_n_1 x[???+?]')),
    'cross': Noun('shape', 'cross', AnchorDmrs.parse('[noun]:_cross_n_1 x[???+?]')),
    'circle': Noun('shape', 'circle', AnchorDmrs.parse('[noun]:_circle_n_1 x[???+?]')),
    'semicircle': Noun('shape', 'semicircle', AnchorDmrs.parse('[noun]:_semicircle_n_1 x[???+?]')),
    'ellipse': Noun('shape', 'ellipse', AnchorDmrs.parse('[noun]:_ellipse_n_1 x[???+?]')),
    'shape': Noun('shape-set', ('square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'), AnchorDmrs.parse('[noun]:_shape_n_1 x[???+?]'))}

DmrsCaptionerComponent.modifiers = {
    'black': Modifier('color', 'black', AnchorDmrs.parse('[mod]:_black_a_1 e? =1=> [noun]:node')),
    'red': Modifier('color', 'red', AnchorDmrs.parse('[mod]:_red_a_1 e? =1=> [noun]:node')),
    'green': Modifier('color', 'green', AnchorDmrs.parse('[mod]:_green_a_1 e? =1=> [noun]:node')),
    'blue': Modifier('color', 'blue', AnchorDmrs.parse('[mod]:_blue_a_1 e? =1=> [noun]:node')),
    'yellow': Modifier('color', 'yellow', AnchorDmrs.parse('[mod]:_yellow_a_1 e? =1=> [noun]:node')),
    'magenta': Modifier('color', 'magenta', AnchorDmrs.parse('[mod]:_magenta_a_1 e? =1=> [noun]:node')),
    'cyan': Modifier('color', 'cyan', AnchorDmrs.parse('[mod]:_cyan_a_1 e? =1=> [noun]:node')),
    'white': Modifier('color', 'white', AnchorDmrs.parse('[mod]:_white_a_1 e? =1=> [noun]:node'))}
