from copy import deepcopy
from pydmrs.components import Pred
from pydmrs.core import Link, ListDmrs
from pydmrs.graphlang.graphlang import parse_graphlang


class Dmrs(ListDmrs):
    # composable dmrs

    __slots__ = ('nodes', 'links', 'index', 'top', 'anchors')

    def __init__(self, nodes=(), links=(), index=None, top=None):
        super(Dmrs, self).__init__(nodes=nodes, links=links, index=index, top=top)
        self.anchors = dict()

    @staticmethod
    def parse(string):
        anchors = dict()
        dmrs = parse_graphlang(string, cls=Dmrs, anchors=anchors)
        dmrs.anchors = anchors
        return dmrs

    def compose(self, other, fusion):
        assert isinstance(other, Dmrs)
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
        for node in self.iter_nodes():
            if type(node.pred) is Pred:
                self.remove_node(node.nodeid)
                # self.remove_links(link for link in self.iter_links() if link.start == node.nodeid or link.end == node.nodeid)
                continue
            if node.sortinfo is None:
                continue
            node.sortinfo = node.sortinfo.__class__(**{key: None if node.sortinfo[key] in ('u', '?') else node.sortinfo[key] for key in node.sortinfo if key != 'cvarsort'})

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
                hcons[index] = labels[link.end]
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
