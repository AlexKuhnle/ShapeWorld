import os
import sys


# dmrs sub-directory
directory = os.path.dirname(os.path.realpath(__file__))

# add pydmrs submodule to Python path
sys.path.insert(1, os.path.join(directory, 'pydmrs'))


from pydmrs.components import Pred, Sortinfo, EventSortinfo, InstanceSortinfo
from pydmrs.core import Link, ListDmrs
from pydmrs.graphlang.graphlang import parse_graphlang
from pydmrs.mapping.paraphrase import paraphrase


def create_sortinfo(cvarsort, features):
    assert len(cvarsort) == 1 and cvarsort != 'i'
    assert isinstance(features, tuple)
    if cvarsort == 'e' and features == ('sf', 'tense', 'mood', 'perf', 'prog'):
        return EventSortinfo
    elif cvarsort == 'x' and features == ('pers', 'num', 'gend', 'ind', 'pt'):
        return InstanceSortinfo
    else:
        return type((cvarsort.upper() + 'Sortinfo'), (Sortinfo,), dict(cvarsort=cvarsort, __slots__=features))


class Dmrs(ListDmrs):
    # composable dmrs

    __slots__ = ('nodes', 'links', 'index', 'top', 'anchors')

    def __init__(self, nodes=(), links=(), index=None, top=None):
        super(Dmrs, self).__init__(nodes=nodes, links=links, index=index, top=top)
        self.anchors = dict()

    @staticmethod
    def parse(string, sortinfo_classes=None, sortinfo_shortforms=None):
        anchors = dict()
        dmrs = parse_graphlang(string, cls=Dmrs, anchors=anchors, sortinfo_classes=sortinfo_classes, sortinfo_shortforms=sortinfo_shortforms)
        dmrs.anchors = anchors
        return dmrs

    def compose(self, other, fusion=None, other_head=False, hierarchy=None):
        assert isinstance(other, Dmrs)
        nodeid_mapping = dict()
        # unify anchors
        if fusion is None:
            for anchor1 in self.anchors:
                for anchor2 in other.anchors:
                    if anchor1 != anchor2:
                        continue
                    node1 = self.anchors[anchor1]
                    node2 = other.anchors[anchor2]
                    node1.unify(node2, hierarchy=hierarchy)
                    nodeid_mapping[node2.nodeid] = node1.nodeid
        else:
            for anchor1, anchor2 in fusion.items():
                node1 = self.anchors[anchor1]
                node2 = other.anchors[anchor2]
                node1.unify(node2, hierarchy=hierarchy)
                nodeid_mapping[node2.nodeid] = node1.nodeid
        # add missing nodes, update node ids
        for node2 in other.iter_nodes():
            nodeid2 = node2.nodeid
            if nodeid2 in nodeid_mapping:
                node2.nodeid = nodeid_mapping[nodeid2]
            else:
                node2.nodeid = None
                nodeid_mapping[nodeid2] = self.add_node(node2)
        # add missing links, update existing links
        links1 = set((link1.start, link1.end) for link1 in self.iter_links())
        for link2 in other.iter_links():
            start = nodeid_mapping[link2.start]
            end = nodeid_mapping[link2.end]
            if (start, end) not in links1:
                link1 = Link(start, end, link2.rargname, link2.post)
                self.add_link(link1)
            if other_head and (start, end) in links1:
                self.remove_link((start, end))
                link1 = Link(start, end, link2.rargname, link2.post)
                self.add_link(link1)
        # update index and top
        if other_head:
            if other.index is None:
                self.index = None
            else:
                self.index = self[other.index.nodeid]
            if other.top is None:
                self.top = None
            else:
                self.top = self[other.top.nodeid]
        # set anchors
        if other_head:
            self.anchors = {anchor: self[node2.nodeid] for anchor, node2 in other.anchors.items()}

    def apply_paraphrases(self, paraphrases):
        return paraphrase(dmrs=self, paraphrases=paraphrases)

    def remove_underspecifications(self):
        for node in list(self.iter_nodes()):
            if type(node.pred) is Pred:
                self.remove_node(node.nodeid)
                self.remove_links(link for link in self.iter_links() if link.start == node.nodeid or link.end == node.nodeid)
                continue
            # TODO: remove underspecification in partially underspecified predicate
            if node.sortinfo is not None:
                node.sortinfo = node.sortinfo.__class__(**{key: None if node.sortinfo[key] in ('u', '?') else node.sortinfo[key] for key in node.sortinfo if key != 'cvarsort'})
            if node.carg == '?':
                node.carg = None

    def get_mrs(self):
        # labels = dict(zip(self, range(1, len(self) + 1)))
        # redirected = []
        quantifiers = dict()
        # labels = dict(zip(self, self))
        labels = {nodeid: [nodeid] for nodeid in self}
        for link in self.iter_links():
            assert isinstance(link.start, int) and isinstance(link.end, int)
            assert link.rargname is not None or link.post == 'EQ'  # ('ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG', 'RSTR', 'BODY', 'L-INDEX', 'R-INDEX', 'L-HNDL', 'R-HNDL')
            assert link.post in ('NEQ', 'EQ', 'H', 'HEQ')
            if link.post == 'EQ':
                # upper, lower = (link.start, link.end) if link.start > link.end else (link.end, link.start)
                # labels[upper] = lower
                labels[link.start].append(link.end)
                labels[link.end].append(link.start)
            elif link.rargname == 'RSTR' and link.post == 'H':
                quantifiers[link.start] = link.end

        # for upper, lower in labels.items():
        #     lower_lower = labels[lower]
        #     while lower_lower != lower:
        #         lower = lower_lower
        #         lower_lower = labels[lower]
        #     labels[upper] = lower
        for nodeid in labels:
            eqs = labels[nodeid]
            if isinstance(eqs, int):
                continue
            lowest = nodeid
            n = 0
            while n < len(eqs):
                lbl = eqs[n]
                if lbl < lowest:
                    lowest = lbl
                for nodeid in labels[lbl]:
                    if nodeid not in eqs:
                        eqs.append(nodeid)
                n += 1
            for lbl in eqs:
                labels[lbl] = lowest

        ordered = sorted(labels.values())
        labels = {nodeid: ordered.index(label) + 1 for nodeid, label in labels.items()}

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
        hcons = {0: labels[self.top.nodeid]}
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
            elempred_string = '[ {} LBL: h{} {}ARG0: {} {}]'.format(predicates[nodeid], labels[nodeid], carg_string, intrinsic_string, args_string)
            elempreds.append(elempred_string)

        top_string = '' if self.top is None else 'TOP: h0 '
        index_string = '' if self.index is None else 'INDEX: {} '.format(variables[self.index.nodeid][0])
        eps_string = '  '.join(elempreds)
        hcons_string = ' '.join('h{} qeq h{}'.format(*qeq) for qeq in hcons.items())
        mrs_string = '[ {}{}RELS: < {} > HCONS: < {} > ICONS: <  > ]'.format(top_string, index_string, eps_string, hcons_string)
        return mrs_string
