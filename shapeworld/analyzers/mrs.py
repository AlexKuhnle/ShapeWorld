from collections import namedtuple
import os
import sys
from shapeworld.realizers.dmrs.pydmrs.pydmrs.core import Link, Node, Dmrs


# add pydmrs submodule to Python path
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'realizers', 'dmrs', 'pydmrs')
sys.path.insert(1, directory)


from pydmrs.components import Pred, GPred, Sortinfo, EventSortinfo, InstanceSortinfo


class Reference(namedtuple('ReferenceTuple', ('sort', 'index'))):
    __slots__ = ()

    def __new__(cls, sort, index):
        assert isinstance(sort, str) and len(sort) == 1 and sort in 'hexipu'
        assert isinstance(index, int) and index >= 0
        return super().__new__(cls, sort, index)

    def __repr__(self):
        return 'Reference({}, {})'.format(repr(self.sort), repr(self.index))

    def __str__(self):
        return '{}{}'.format(*self)

    def is_handle(self):
        return self.sort == 'h'

    def is_event(self):
        return self.sort == 'e'

    def is_instance(self):
        return self.sort == 'x'

    def is_sortinfo(self):
        return self.sort == 'i'

    def is_not_handle(self):
        return self.sort in 'exi'

    def is_not_event(self):
        return self.sort in 'hxp'

    def is_underspecified(self):
        return self.sort == 'u'


class Variable(namedtuple('VariableTuple', ('ref', 'sortinfo'))):
    __slots__ = ()

    def __new__(cls, ref, sortinfo):
        assert isinstance(ref, Reference)
        assert isinstance(sortinfo, Sortinfo)
        assert ref.sort == sortinfo.cvarsort
        return super().__new__(cls, ref, sortinfo)

    def __repr__(self):
        return 'Variable({}, {})'.format(repr(self.ref), repr(self.sortinfo))

    def __str__(self):
        return '{} [ {} {}]'.format(self.ref, self.ref.sort, ''.join('{}: {} '.format(key.upper().replace('_DASH_', '-').replace('_DOT_', '.'), value.lower()) for key, value in self.sortinfo.iter_specified() if key != 'cvarsort' and value is not None))


class ElemPred(Node):

    __slots__ = ('nodeid', 'mrs', 'label', 'pred', 'intrinsic', 'carg', 'args', 'cfrom', 'cto')

    def __init__(self, nodeid=None, pred=None, sortinfo=None, cfrom=None, cto=None, surface=None, base=None, carg=None, args=None, label=None, intrinsic=None):
        # no sortinfo, only when adding
        assert nodeid is None or isinstance(nodeid, int)
        assert label is None or (isinstance(label, Reference) and label.is_handle())
        assert pred is None or isinstance(pred, Pred)
        assert intrinsic is None or (isinstance(intrinsic, Reference) and intrinsic.is_not_handle())
        assert carg is None or isinstance(carg, str)
        assert args is None or (isinstance(args, dict) and all(isinstance(role, str) and isinstance(arg, Reference) for role, arg in args.items()))
        self.nodeid = nodeid
        self.mrs = None
        self.label = label
        self.pred = pred
        self.intrinsic = intrinsic
        self.carg = carg[1:-1] if (carg is not None and carg[0] == '"' and carg[-1] == '"') else carg
        self.args = {} if args is None else {role.lower(): arg for role, arg in args.items()}
        self.cfrom = cfrom
        self.cto = cto
        if sortinfo is not None:
            assert isinstance(sortinfo, Sortinfo)
            assert intrinsic is None
            self.intrinsic = sortinfo

    def __repr__(self):
        return 'ElemPred({}, {}, {}, {}, {}, {})'.format(repr(self.label), repr(self.nodeid), repr(self.pred), repr(self.carg), repr(self.intrinsic), repr(self.args))

    def __str__(self):
        if self.cfrom is None or self.cto is None:
            span = ''
        else:
            span = '<{}:{}>'.format(self.cfrom, self.cto)
        if self.is_quantifier():
            var = self.intrinsic
            if var.is_event():  # TODO: Hack for Chinese
                var = self.mrs.head_ep(label=self.mrs.hcons[self.mrs.var_eps[var].args['arg1']]).intrinsic
        elif self.is_icon():
            var = Variable(self.intrinsic, EventSortinfo('prop', None, None, None, None))
        else:
            var = self.mrs.vars[self.intrinsic]
        args_str = ''.join('{}: {} '.format(role.upper(), var if role == 'arg0' else self.args[role]) for role in ('arg1', 'arg2', 'arg3', 'arg4', 'arg', 'rstr', 'body', 'l-index', 'r-index', 'l-hndl', 'r-hndl') if role in self.args)
        if self.carg is None:
            return '[ {}_rel{} LBL: {} ARG0: {} {}]'.format(self.pred, span, self.label, var, args_str)
        else:
            return '[ {}_rel{} LBL: {} CARG: "{}" ARG0: {} {}]'.format(self.pred, span, self.label, self.carg, var, args_str)

    @property
    def sortinfo(self):
        if self.is_quantifier():
            return None
        elif self.is_icon():
            return EventSortinfo('prop', None, None, None, None)
        return self.mrs.vars[self.intrinsic].sortinfo

    @property
    def surface(self):
        return None

    @property
    def base(self):
        return None

    def is_quantifier(self):
        return self.args and all(role in ('rstr', 'body') for role in self.args)

    def is_coordination(self):
        return self.args and all(role in ('l-index', 'r-index', 'l-hndl', 'r-hndl') for role in self.args)

    def is_unknown(self):
        return isinstance(self.pred, GPred) and self.pred.name == 'unknown' and self.args and all(role == 'arg' for role in self.args)

    def is_icon(self):
        return isinstance(self.pred, GPred) and self.pred.name[-2:] == '_d' and self.args and all(role in ('arg1', 'arg2') for role in self.args)

    def is_normal(self):
        return all('arg{}'.format(n + 1) in self.args for n in range(len(self.args)))

    def valid(self):
        if self.is_quantifier():
            assert self.intrinsic.is_instance()
            assert all(arg.is_handle() for arg in self.args.values())
        elif self.is_coordination():
            if isinstance(self.pred, GPred) and self.pred.name == 'fw_seq':  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                return True
            else:
                if 'l-index' in self.args and self.args['l-index'].is_sortinfo() and 'l-hndl' in self.args and self.args['l-hndl'].is_underspecified():
                    assert 'r-hndl' in self.args and self.args['r-hndl'].is_handle() and 'r-index' in self.args and self.args['r-index'] == self.intrinsic.sort and 'r-hndl' in self.args and self.args['r-hndl'].is_handle()
                else:
                    assert all(arg.sort == self.intrinsic.sort for role, arg in self.args.items() if role in ('l-index', 'r-index')), 'Invalid ElemPred.'
                    assert all(arg.is_handle() for role, arg in self.args.items() if role in ('l-hndl', 'r-hndl'))
        elif self.is_unknown():
            assert self.intrinsic.is_event()
        elif self.is_icon():
            assert self.intrinsic.is_event()
        elif self.is_normal():
            assert len(self.args) <= 4
        else:
            assert False
        return True


class Mrs(Dmrs):

    Node = ElemPred

    def __init__(self, nodes=(), links=(), cfrom=None, cto=None, surface=None, ident=None, index=None, top=None):
        self.vars = {}
        # each EP is labeled with a handle
        # within a set of equally-labeled EPs there is at most one EP associated with a variable as its intrinsic variable
        self.eps = {}  # {h-ref: {e/x/i-ref: ep}}
        # each variable is associated with a unique non-quantifier EP as its intrinsic variable
        self.var_eps = {}  # {e/x/i-ref: ep}
        # every EP without associated intrinsic variable is a quantifier EP
        # every variable is associated with at most one quantifier EP as its intrinsic variable
        self.quant_eps = {}  # {x-ref: ep}
        self.hcons = {}  # {hole: label}
        self.icons = []
        # for icons represented by nodes: [(hole, label)]   (same length as icons, if existent)
        self.icon_hcons = []
        self.top_handle = Reference('h', 0)
        self.index_ref = None
        # DMRS constructor
        self.add_nodes(nodes)
        self.add_links(links)
        if isinstance(top, Node):
            self.top = top
            self.hcons[self.top_handle] = self.top.label
        elif isinstance(top, int):
            self.top = self[top]
            self.hcons[self.top_handle] = self.top.label
        else:
            self.top = None
        if isinstance(index, Node):
            self.index = index
            self.index_ref = self.index.intrinsic
        elif isinstance(index, int):
            self.index = self[index]
            self.index_ref = self.index.intrinsic
        else:
            self.index = None

    @property
    def cfrom(self):
        return None

    @property
    def cto(self):
        return None

    @property
    def surface(self):
        return None

    @property
    def ident(self):
        return None

    # ===============================|  Basics  |==============================

    def __str__(self):
        return '[ {}{}RELS: < {} > HCONS: < {} > {}]'.format(
             '' if self.top_handle is None else 'TOP: {} '.format(self.top_handle),
             '' if self.index_ref is None else 'INDEX: {} '.format(self.index_ref),
             '  '.join(str(ep) for ep in self.iter_eps(icon_mode=True)),
             ' '.join('{} qeq {}'.format(hole, label) for hole, label in self.iter_hcons(icon_mode=True)),
             '' if self.icon_hcons else 'ICONS: < {} > '.format(' '.join('{} {} {}'.format(*icon) for icon in self.iter_icons(icon_mode=True))))

    def pretty(self):
        return '[ {}{}\n  RELS: <\n    {} >\n  HCONS: < {} > {}]'.format(
            '' if self.top_handle is None else 'TOP: {} '.format(self.top_handle),
            '' if self.index_ref is None else 'INDEX: {} '.format(self.index_ref),
            '\n    '.join(str(ep) for ep in self.iter_eps(icon_mode=True)),
            ' '.join('{} qeq {}'.format(hole, label) for hole, label in self.iter_hcons(icon_mode=True)),
            '' if self.icon_hcons else '\n  ICONS: < {} > '.format(' '.join('{} {} {}'.format(*icon) for icon in self.iter_icons(icon_mode=True))))

    def valid(self):
        return all(ep.valid() for ep in self.iter_eps())
        refs = set()
        for ep in self.iter_eps():
            if ep.intrinsic in refs:
                refs.remove(ep.intrinsic)
            else:
                refs.add(ep.intrinsic)
            if ep.is_quantifier():
                assert ep.intrinsic in self.quant_eps
                assert ep.intrinsic == self.quant_eps[ep.intrinsic].intrinsic
                assert ep.intrinsic in self.var_eps
                assert ep.intrinsic in self.vars
            else:
                assert ep.intrinsic in self.var_eps
                assert ep.intrinsic == self.var_eps[ep.intrinsic].intrinsic
                assert ep.intrinsic in self.vars
                assert ep.intrinsic == self.vars[ep.intrinsic].ref
                if ep.intrinsic.is_instance():
                    assert ep.intrinsic in self.quant_eps
        for ref in refs:
            assert ref.is_event()

    def is_canonical(self):
        quant_labels = set()
        for ep in self.quant_eps.values():
            quant_labels.add(ep.label)
            assert len(self.eps[ep.label]) == 1, 'No quantifier EP is involved in an EP conjunction.'
            assert 'body' not in ep.args or ep.args['body'] not in self.hcons, 'The body hole of no quantifier EP is involved in any constraint.'
            assert ep.label not in self.hcons.values(), 'The label of no quantifier EP is involved in any constraint.'
        for label, eps in self.eps.items():
            if label in quant_labels:
                continue
            assert label in self.hcons.values(), 'Every other label occurs in exactly one constraint.'
            assert all(not arg.is_handle() or arg in self.eps or arg in self.hcons for arg in ep.args.values()), 'Every other hole occurs in exactly one constraint.'
        return True

    def free_index(self):
        max_index = self.top_handle.index
        for ep in self.iter_eps():
            max_index = max(max_index, ep.label.index, -1 if ep.intrinsic is None else ep.intrinsic.index, *[arg.index for arg in ep.args.values()])
        return max_index + 1

    # =============================|  Iterators  |=============================

    def iter_eps(self, icon_mode=False):
        if icon_mode and self.icon_hcons:
            icon_hcons_iter = iter(self.icon_hcons)
            eps = list(self.iter_nodes())
            free_index = self.free_index()
            for ref1, icon, ref2 in self.icons:
                _, label = next(icon_hcons_iter)
                eps.append(ElemPred(label=label, pred=GPred(icon + '_d'), intrinsic=Reference('e', free_index), args=[('arg1', ref1), ('arg2', ref2)]))
                free_index += 1
            return iter(eps)
        else:
            return (ep for eps in self.eps.values() for ep in eps.values())

    def iter_hcons(self, icon_mode=False):
        if icon_mode and self.icon_hcons:
            return iter(list(self.hcons.items()) + self.icon_hcons)
        else:
            return iter(self.hcons.items())

    def iter_icons(self, icon_mode=False):
        if icon_mode and self.icon_hcons:
            return iter(())
        else:
            return iter(self.icons)

    # ================================|  Head  |===============================

    def find(self, source_ep, target_ep):
        distance = len(self)
        for arg in source_ep.args.values():
            if arg == target_ep.intrinsic:
                return 1
            else:
                if arg.is_handle():
                    if arg in self.eps:
                        distance = min(distance, *[1 + self.find(ep, target_ep) for ep in self.eps[arg].values() if ep != source_ep])
                    elif arg in self.hcons:
                        for label in self.eps:
                            if self.hcons[arg] == label:
                                distance = min(distance, *[1 + self.find(ep, target_ep) for ep in self.eps[label].values() if ep != source_ep])
                elif arg in self.var_eps:
                    distance = min(distance, 1 + self.find(self.var_eps[arg], target_ep))
        return distance

    def head_ep(self, label):
        assert label in self.eps
        if len(self.eps[label]) == 1:  # one unique EP labeled with this handle
            return next(iter(self.eps[label].values()))
        # head_refs = set(self.eps[label])
        # only consider EPs with no variable argument associated with other EPs in the set
        head_eps = [ep for ep in self.eps[label].values()]  # if all(arg not in head_refs for arg in ep.args.values())]  # second one for "nearly all", but potentially more??????????????????????? --> test!!!!!!!!!!!!!!!!!!!!
        assert head_eps
        if len(head_eps) == 1:
            return head_eps.pop()
        head_eps = [target_ep for target_ep in head_eps if any(arg in self.eps or arg in self.hcons or arg in self.vars for arg in target_ep.args.values()) or any(self.find(ep, target_ep) < len(self) for ep in head_eps)]
        assert head_eps, 'Weird equal label set: {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(label)
        if len(head_eps) == 1:
            return head_eps.pop()
        # look for distance
        head_ep1 = None
        for target_ep in head_eps:
            # for ep in head_eps:
                # if ep.intrinsic != target_ep.intrinsic and self.find(ep, target_ep) < len(self):
                #     if head_ep1 is not None:
                #         print(head_ep1, target_ep)
                #     assert head_ep1 is None or head_ep1.intrinsic == target_ep.intrinsic, 'Ambiguous transitive head heuristic for label {}: {}'.format(label, self)
            if all(self.find(ep, target_ep) < len(self) for ep in head_eps if ep.intrinsic != target_ep.intrinsic):
                assert head_ep1 is None, 'Ambiguous transitive head heuristic for label {}: {}'.format(label, self)
                head_ep1 = target_ep
        if head_ep1 is not None:  # !!!!!!!!!
            return head_ep1
        head_ep2 = None
        for l, eps in self.eps.items():  # incomings
            if l != label:
                for ep in eps.values():
                    for arg in ep.args.values():
                        for e in head_eps:
                            if arg == e.intrinsic:
                                assert head_ep2 is None or head_ep2.intrinsic == e.intrinsic, 'Problem1: {} {}'.format('None' if head_ep2 is None else head_ep2.intrinsic, e.intrinsic)
                                head_ep2 = e
        head_ep3 = None
        for ep in head_eps:
            if any(arg.is_handle() for arg in ep.args.values()):
                # assert all(arg.is_handle() for _, arg in ep.args)
                assert head_ep3 is None, 'Problem3: {}'.format(self)
                head_ep3 = ep
        head_ep4 = None
        if len(head_eps) > 2:
            for ep in head_eps:
                for target_ep in head_eps:
                    if ep.intrinsic != target_ep.intrinsic and self.find(ep, target_ep) < len(self):
                        assert head_ep4 is None or head_ep4.intrinsic == ep.intrinsic, 'Ambiguous head heuristic for label {}: {}'.format(label, self)
                        head_ep4 = ep
        # if head_ep2 is None:
        #     assert self.top_handle is not None, 'Transitive head heuristic failed for label {} and top handle is not defined: {}'.format(label, self)
        #     assert label == self.hcons[self.top_handle], 'Transitive head heuristic failed for label {} and label is not qeq top handle: {}'.format(label, self)
        #     assert self.index_ref is not None, 'Transitive head heuristic failed for label {} and index ref is not defined: {}'.format(label, self)
        #     assert self.index_ref in (ep.intrinsic for ep in head_eps), 'Transitive head heuristic failed for label {} and index ref is not potential head: {}'.format(label, self)
        #     head_ep2 = self.eps[label][self.index_ref]
        # else:
        #     assert head_ep2.intrinsic == self.index_ref, 'Problem2: {} {}'.format(head_ep2.intrinsic, self.index_ref)
        assert head_ep1 is not None or head_ep2 is not None or head_ep3 is not None or head_ep4 is not None, 'Heuristics failed: {}, {}'.format(label, self)
        assert (head_ep1 is not None) + (head_ep2 is not None) + (head_ep3 is not None) + (head_ep4 is not None) == 1, 'Problem4: {} {} {} {} {}'.format((head_ep1 is not None) + (head_ep2 is not None) + (head_ep3 is not None) + (head_ep4 is not None), head_ep1, head_ep2, head_ep3, head_ep4)
        return head_ep1 or head_ep2 or head_ep3 or head_ep4

        # Ann's version
        # the head EPs in an equally-labeled set of EPS are defined as those 
        # which have no variable arguments associated with EPs in the set (no 
        # outgoing variable links to EPs in the set)
        # return [ep for ep in self.eps[handle].values() if not any(arg in self.eps[handle] for _, arg in ep.args)]

    # ================================|  DMRS  |===============================

    def __len__(self):
        return sum(len(eps) for eps in self.eps.values())

    def __iter__(self):
        return (ep.nodeid for ep in self.iter_nodes())

    def __getitem__(self, nodeid):
        for ep in self.iter_nodes():
            if ep.nodeid == nodeid:
                return ep
        raise KeyError

    def iter_nodes(self):
        # each EP becomes a node in the DMRS graph
        return (ep for eps in self.eps.values() for ep in eps.values())

    def iter_links(self):
        equal_check = set()
        for label_start in self.eps:
            for ref_start in self.eps[label_start]:
                ep_start = self.eps[label_start][ref_start]
                start = ep_start.nodeid
                for role, arg in ep_start.args.items():
                    post = None
                    if arg.is_event() or arg.is_instance():  # variable link
                        # variable links are pointing to the non-quantifier EP associated with the argument
                        ref_end = arg
                        if ref_end not in self.var_eps:
                            # assert ref_end.sort not in 'ex'
                            continue
                        label_end = self.var_eps[ref_end].label
                        end = self.eps[label_end][ref_end].nodeid
                        if label_start == label_end:  # equal labels
                            # if the labels of the two EPs are equal, the link post is EQ
                            post = 'EQ'
                            equal_check.add(ref_start)
                            equal_check.add(ref_end)
                        else:
                            # if the labels of the two EPs are not equal, the link post is NEQ
                            post = 'NEQ'
                        yield Link(start, end, role, post)

                    elif arg.is_handle():  # handle link (hole argument)
                        # handle links are pointing to an EP labeled with this or a qeq handle
                        end = None
                        if arg in self.eps:
                            # if an EP labeled with the argument handle exists, the link post is HEQ
                            post = 'HEQ'
                            label_end = arg
                        else:
                            for label in self.eps:
                                if arg in self.hcons and self.hcons[arg] == label:
                                    # if an EP labeled with a handle qeq to the argument handle exists, the link post is H
                                    assert post is None
                                    post = 'H'
                                    label_end = label
                            if post is None:  # no link if no EP is found
                                continue
                        if ep_start.intrinsic in self.eps[label_end]:
                            assert post != 'HEQ'
                            ref_end = ep_start.intrinsic
                            end = self.eps[label_end][ref_end].nodeid
                        else:
                            # for ep_end in self.head_eps(handle_end): [if head unique]
                            ep_end = self.head_ep(label_end)
                            ref_end = ep_end.intrinsic
                            end = ep_end.nodeid
                        if label_start == label_end:
                            equal_check.add(ref_start)
                            equal_check.add(ref_end)
                        yield Link(start, end, role, post)

                    else:
                        assert arg not in self.var_eps and arg not in self.eps and arg not in self.hcons

        for handle, eps in self.eps.items():
            if len(eps) == 1:
                continue
            head_ep = self.head_ep(handle)
            not_covered = [ep for ep in eps.values() if ep.intrinsic not in equal_check]
            end = head_ep.nodeid
            if head_ep.intrinsic in equal_check:
                for ep in not_covered:
                    start = ep.nodeid
                    yield Link(start, end, None, 'EQ')
            else:
                # arbitrary decision !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                equals = [ep for ep in eps.values() if ep.intrinsic in equal_check]
                if equals:
                    start = equals[0].nodeid
                    yield Link(start, end, None, 'EQ')
                for ep in not_covered:
                    if ep.intrinsic != head_ep.intrinsic:
                        start = ep.nodeid
                        yield Link(start, end, None, 'EQ')
                # for ref in equal_check:
                #     start = eps[ref].nodeid
                #     yield Link(start, end, None, 'EQ')

    @property
    def nodes(self):
        return list(self.iter_nodes())

    @property
    def links(self):
        return list(self.iter_links())

    def free_nodeid(self):
        if len(self):
            return max(self) + 1
        else:
            return 1

    # ================================|  Add  |================================

    def add_var(self, var):
        assert var.ref not in self.vars, 'Variable {} is already defined. {}'.format(var.ref, self)
        self.vars[var.ref] = var

    def add_node(self, node):
        node.mrs = self
        if node.nodeid is None and isinstance(node.intrinsic, Reference):
            # created as EP
            node.nodeid = self.free_nodeid()
            if node.is_quantifier():
                assert node.intrinsic not in self.quant_eps
                self.quant_eps[node.intrinsic] = node
            else:
                assert node.intrinsic in self.vars
                assert node.intrinsic not in self.var_eps
                self.var_eps[node.intrinsic] = node
        elif node.label is None and (node.intrinsic is None or isinstance(node.intrinsic, Sortinfo)):
            # created as node
            assert node.nodeid != 0
            #node.nodeid = -node.nodeid if node.nodeid > 0 else node.nodeid
            assert node.nodeid not in self
            free_index = self.free_index()
            node.label = Reference('h', free_index)
            if node.intrinsic is not None:
                var = Variable(Reference(node.intrinsic.cvarsort, free_index + 1), node.intrinsic)
                self.add_var(var)
                node.intrinsic = var.ref
                assert node.intrinsic not in self.var_eps
                self.var_eps[node.intrinsic] = node
        else:
            assert False
        if node.label not in self.eps:
            self.eps[node.label] = {node.intrinsic: node}
        else:
            assert node.intrinsic not in self.eps[node.label]
            self.eps[node.label][node.intrinsic] = node

    def add_link(self, link):
        ep_start = self[link.start]
        ep_end = self[link.end]
        if link.rargname is None:
            assert link.post == 'EQ'
            label = Reference('h', self.free_index())
            self.change_ref(ep_start.label, label)
            self.change_ref(ep_end.label, label)
        else:
            assert link.rargname in ('ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG', 'RSTR', 'BODY', 'L-INDEX', 'R-INDEX', 'L-HNDL', 'R-HNDL')
            if link.post == 'NEQ':
                assert ep_start.label != ep_end.label
                ep_start.args[link.rargname.lower()] = ep_end.intrinsic
            elif link.post == 'EQ':
                label = Reference('h', self.free_index())
                self.change_ref(ep_start.label, label)
                self.change_ref(ep_end.label, label)
                ep_start.args[link.rargname.lower()] = ep_end.intrinsic
            elif link.post == 'H':
                handle = Reference('h', self.free_index())
                ep_start.args[link.rargname.lower()] = handle
                self.add_hcon(handle, ep_end.label)
                if link.rargname == 'RSTR':
                    assert ep_start.intrinsic is None
                    ep_start.intrinsic = ep_end.intrinsic
                    assert ep_start.intrinsic not in self.quant_eps
                    self.quant_eps[ep_start.intrinsic] = ep_start
            elif link.post == 'HEQ':
                ep_start.args[link.rargname.lower()] = ep_end.label
            else:
                assert False

    def add_hcon(self, hole, label, icon_label=False):
        assert hole not in self.eps
        assert hole not in self.hcons
        assert hole not in self.icon_hcons
        if icon_label:
            self.icon_hcons.append((hole, label))
            assert len(self.icon_hcons) == len(self.icons)
        else:
            assert label in self.eps, 'Label {} of hcon must be associated with an EP. {}'.format(label, self)
            self.hcons[hole] = label

    def add_icon(self, ref1, icon, ref2):  # TODO
        # assert ref1.is_not_handle() and ref2.is_not_handle()    # p occurs
        assert icon in ['focus', 'parg', 'topic'], 'Invalid icon: {}'.format(icon)
        self.icons.append((ref1, icon, ref2))

    def change_ref(self, old_ref, new_ref):
        if self.top_handle == old_ref:
            self.top_handle = new_ref
        if self.index_ref == old_ref:
            self.index_ref = new_ref
        if old_ref in self.eps:
            if new_ref not in self.eps:
                self.eps[new_ref] = {}
            for ep in self.eps.pop(old_ref).values():
                ep.label = new_ref
                assert ep.intrinsic not in self.eps[new_ref]
                self.eps[new_ref][ep.intrinsic] = ep
        if old_ref in self.quant_eps:
            assert new_ref not in self.quant_eps
            self.quant_eps[new_ref] = self.quant_eps.pop(old_ref)
            self.quant_eps[new_ref].intrinsic = new_ref
            assert new_ref not in self.eps[self.quant_eps[new_ref].label]
            self.eps[self.quant_eps[new_ref].label][new_ref] = self.eps[self.quant_eps[new_ref].label].pop(old_ref)
        if old_ref in self.var_eps:
            assert new_ref not in self.var_eps
            self.var_eps[new_ref] = self.var_eps.pop(old_ref)
            self.var_eps[new_ref].intrinsic = new_ref
            assert new_ref not in self.eps[self.var_eps[new_ref].label]
            self.eps[self.var_eps[new_ref].label][new_ref] = self.eps[self.var_eps[new_ref].label].pop(old_ref)
            assert new_ref not in self.vars
            assert old_ref.is_sortinfo()
            self.vars.pop(old_ref)
            if new_ref.is_event():
                self.vars[new_ref] = Variable(new_ref, EventSortinfo(None, None, None, None, None))
            elif new_ref.is_instance():
                self.vars[new_ref] = Variable(new_ref, InstanceSortinfo(None, None, None, None, None))
            else:
                assert False
        for eps in self.eps.values():
            for ep in eps.values():
                for i, (role, arg) in enumerate(ep.args.items()):
                    if arg == old_ref:
                        ep.args[role] = new_ref
        for hole, label in self.hcons.items():
            if hole == old_ref:
                self.hcons[new_ref] = label
            if label == old_ref:
                self.hcons[hole] = new_ref
        for i, (ref1, icon, ref2) in enumerate(self.icons):
            if ref1 == old_ref:
                self.icons[i] = (new_ref, icon, ref2)
            if ref2 == old_ref:
                self.icons[i] = (ref1, icon, new_ref)
        for i, (hole, label) in enumerate(self.icon_hcons):
            if hole == old_ref:
                self.icon_hcons[i] = (new_ref, label)
            if label == old_ref:
                self.icon_hcons[i] = (hole, new_ref)

    def iter_outgoing(self, nodeid): raise NotImplementedError
    def iter_incoming(self, nodeid): raise NotImplementedError
    def remove_node(self, nodeid): raise NotImplementedError
    def renumber_node(self, old_id, new_id): raise NotImplementedError
    def remove_link(self, link): raise NotImplementedError

    @staticmethod
    def from_string(string):
        from shapeworld.analyzers.mrs_load import read_mrs
        return read_mrs(string)
