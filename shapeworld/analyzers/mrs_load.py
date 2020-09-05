import os
import sys
from shapeworld.analyzers.mrs import Reference, Variable, ElemPred, Mrs


# add pydmrs submodule to Python path
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'realizers', 'dmrs', 'pydmrs')
print(directory)
sys.path.insert(1, directory)


from pydmrs.components import Pred, GPred, RealPred, Sortinfo, EventSortinfo, InstanceSortinfo


def find_next(string, start=None, end=None, whitespace=False):
    if start is None or start < 0:
        start = 0
    if end is None or end < 0:
        end = len(string)
    for i in range(start, end):
        if (string[i] == ' ') is whitespace:
            return i
    return -1


def find_previous(string, start=None, end=None, whitespace=False):
    if start is None or start < 0:
        start = 0
    if end is None or end < 0:
        end = len(string)
    for i in range(end-1, start-1, -1):
        if (string[i] == ' ') is whitespace:
            return i
    return -1


bracket_mapping = {'(': ')', '[': ']', '{': '}', '<': '>', '"': '"', '\'': '\'', '`': '\''}  # , '“': '”'}
quote_brackets = '"\'`'  # “'


# robust quote
def find_substring(string, substring, start=None, end=None, exclude_brackets='', exclude_quotes='', allow_escape=False):
    if start is None or start < 0:
        start = 0
    if end is None or end < 0:
        end = len(string)
    assert all(b in bracket_mapping for b in exclude_brackets), 'Invalid excluded bracket type.'
    assert all(b in quote_brackets for b in exclude_quotes), 'Invalid excluded quote type.'
    opening_brackets = ''.join(b for b in bracket_mapping if b not in exclude_brackets)
    closing_brackets = ''.join(b2 for b1, b2 in bracket_mapping.items() if b1 not in exclude_brackets)
    opening_quotes = ''.join(q for q in quote_brackets if q not in exclude_quotes)
    #closing_quotes = ''.join(bracket_mappinq for q in quote_brackets if q not in exclude_quotes)
    length = len(substring)
    brackets = []
    quote = False
    escape = False
    for i in range(start, end - length + 1):
        if escape:
            escape = string[i] == '\\'
        else:
            if not brackets and string[i:i+length] == substring:
                return i
            elif brackets and string[i] == brackets[-1]:
                brackets.pop()
                if quote:
                    quote = False
            elif not quote and string[i] in opening_brackets:
                brackets.append(bracket_mapping[string[i]])
                if string[i] in opening_quotes:
                    quote = True
            elif not quote and string[i] in closing_brackets:
                assert False, '{}::   {}'.format(string[i-10:i+11], string[start:end])
            elif allow_escape and string[i] == '\\':
                escape = True
    return -1


# ('vice+versa',



def _read_attributes(string, not_lower=(), keys=()):
    attributes = dict()
    l = 0
    r = find_substring(string, ':', start=l)
    while r != -1:
        l = find_next(string, start=l)
        key = string[l:r].lower()
        assert not keys or key in keys
        l = find_next(string, start=r+1)
        r = find_substring(string, ':', start=l)
        if r == -1:
            m = -1
        else:
            m = find_previous(string, start=l, end=r, whitespace=True)
        m = find_previous(string, start=l, end=m) + 1
        value = string[l:m]
        attributes[key] = value if key in not_lower else value.lower()
        l = m
    return attributes


def _read_reference(string, vs, rs, replace=True):
    if '[' in string:
        ref, var = _read_variable(string, vs, rs, replace=replace)
        if var is not None:
            vs[ref] = var
        return ref
    assert string[0] in 'hexipu' and string[1:].isdigit()
    index = int(string[1:])
    ref = Reference(string[0], index)
    if replace:
        ref = rs.get(ref, ref)
    return ref


def _read_variable(string, vs, rs, replace=True):
    if '[' not in string:
        ref = _read_reference(string, vs, rs, replace=replace)
        if replace:
            ref = rs.get(ref, ref)
        if ref in vs:
            return ref, vs.pop(ref)
        elif ref.sort == 'e':
            return ref, Variable(ref, EventSortinfo(None, None, None, None, None))
        elif ref.sort == 'x':
            return ref, Variable(ref, InstanceSortinfo(None, None, None, None, None))
        else:
            return ref, Variable(ref, Sortinfo())

    assert string[-1] == ']'
    r = string.index('[')
    ref = _read_reference(string[:find_previous(string, end=r)+1], vs, rs, replace=replace)
    if replace:
        ref = rs.get(ref, ref)
    l = find_next(string, start=r+1)
    assert replace or string[l] == ref.sort
    l = find_next(string, start=l+1)
    r = find_previous(string, end=len(string)-1) + 1
    attributes = _read_attributes(string[l:r])
    if 'prontype' in attributes:
        assert 'pt' not in attributes
        attributes['pt'] = attributes.pop('prontype')
    if any(attr in ('sf', 'tense', 'mood', 'perf', 'prog') for attr in attributes):
        assert not any(attr in ('pers', 'num', 'gend', 'ind', 'pt') for attr in attributes)
        sf = attributes.pop('sf', None)
        tense = attributes.pop('tense', None)
        mood = attributes.pop('mood', None)
        perf = attributes.pop('perf', None)
        prog = attributes.pop('prog', None)
        assert sf in ('u', 'comm', 'prop', 'ques', 'prop-or-ques', 'prop-or-like'), 'Sentence force'
        assert tense in (None, 'u', 'untensed', 'tensed', 'pres', 'past', 'fut'), 'Tense'
        assert mood in (None, 'u', 'indicative', 'subjunctive'), 'Mood'
        assert perf in (None, 'u', '+', '-', 'plus', 'minus'), 'Perfect'
        assert prog in (None, 'u', '+', '-', 'plus', 'minus', 'bool'), 'Progressive'
        sortinfo = EventSortinfo(sf, tense, mood, perf, prog)
    elif any(attr in ('pers', 'num', 'gend', 'ind', 'pt') for attr in attributes):
        assert not any(attr in ('sf', 'tense', 'mood', 'perf', 'prog') for attr in attributes)
        pers = attributes.pop('pers', None)
        num = attributes.pop('num', None)
        gend = attributes.pop('gend', None)
        ind = attributes.pop('ind', None)
        pt = attributes.pop('pt', None)
        assert pers in (None, 'u', '1', '2', '3', '1-or-3'), 'Person'
        assert num in (None, 'u', 'sg', 'pl'), 'Number'
        assert gend in (None, 'u', 'f', 'm', 'n', 'm-or-f'), 'Gender'
        assert ind in (None, 'u', '+', '-', 'plus', 'minus'), 'Individuated'
        assert pt in (None, 'u', 'notpro', 'std', 'std_pron', 'zero', 'zero_pron', 'refl'), 'Pronoun type'
        sortinfo = InstanceSortinfo(pers, num, gend, ind, pt)
    else:
        if ref.sort == 'e':
            sortinfo = EventSortinfo(None, None, None, None, None)
        elif ref.sort == 'x':
            sortinfo = InstanceSortinfo(None, None, None, None, None)
        else:
            sortinfo = Sortinfo()
    assert not attributes, 'Invalid variable attributes: {}'.format(attributes)
    if ref.sort != sortinfo.cvarsort:
        if replace:
            var = None
        else:
            replace_ref = Reference(sortinfo.cvarsort, ref.index)
            rs[ref] = replace_ref
            var = Variable(replace_ref, sortinfo)
    else:
        var = Variable(ref, sortinfo)
    return ref, var


def _get_reference_replacements(string):
    vs = {}
    rs = {}
    references = set()
    l = find_next(string, start=1)
    r = string.index('RELS:', l)
    r = find_previous(string, start=l, end=r)
    args = _read_attributes(string[l:r+1])
    if 'top' in args:
        references.add(_read_reference(args.pop('top'), vs, rs, replace=False))
    if 'ltop' in args:
        references.add(_read_reference(args.pop('ltop'), vs, rs, replace=False))
    if 'index' in args:
        index_ref = _read_reference(args.pop('index'), vs, rs, replace=False)
        references.add(index_ref)
    l = string.index('RELS:', r)
    l_ep = string.index('<', l)
    r_ep = find_substring(string, '>', start=l_ep+1)

    while l_ep < r_ep:
        l = string.index('[', l_ep)
        r = find_substring(string, ']', start=l+1) + 1
        l_ep = find_next(string, start=r)

        m = string.find('<', l, r_ep)
        if m >= 0:
            m = string.index('>', m, r_ep)
        else:
            m = string.index(' ', l)
        l = find_next(string, start=m+1, whitespace=True)
        r = find_previous(string, start=l, end=r-1) + 1
        args = _read_attributes(string[l:r], not_lower=('carg',))
        args.pop('carg', None)
        references.add(_read_reference(args.pop('lbl'), vs, rs, replace=False))
        intrinsic, _ = _read_variable(args.pop('arg0'), vs, rs, replace=False)
        references.add(intrinsic)
        if all(role in ('rstr', 'body') for role in args):
            if intrinsic.sort == 'i':
                new_intrinsic = Reference('x', intrinsic.index)
                rs[intrinsic] = new_intrinsic
        for role in args:
            references.add(_read_reference(args[role], vs, rs, replace=False))
    assert len(set(ref.index for ref in references)) == len(references)
    for ref in vs:  # either non-existent index ref or non-intrinsic argument
        if ref != index_ref:
            rs.pop(ref, None)
    return rs


def _read_elempred(string, vs, rs):
    assert string[0] == '[' and string[-1] == ']'
    l = 1
    while string[l] == ' ':
        l += 1
    m = string.find('<', l)
    if m >= 0:
        r = string.index('>', m)
        c = string.find(':', m, r)
        if c >= 0:
            cfrom = int(string[m+1:c])
            cto = int(string[c+1:r])
        else:
            assert m + 1 == r
            cfrom = None
            cto = None
    else:
        m = r = string.index(' ', l)
        cfrom = None
        cto = None
    if string[m-15:m] == '_u_unknown_rel"':
        assert string[l:l+2] == '"_'
        slash = string.index('/', l + 2, m - 15)
        pos = {'FW': 'u', 'JJ': 'a', 'NN': 'n', 'NNS': 'n', 'RB': 'a', 'VB': 'v', 'VBP': 'v', 'VBG': 'v', 'VBN': 'v'}  # ?????????????????????????????????????
        assert string[slash+1:m-15] in pos, 'Invalid unknown word POS: {}'.format(string[slash+1:m-15])
        pred = Pred.from_string(string[l+1:slash] + '_' + pos[string[slash+1:m-15]] + string[m-13:m-1])
    else:
        assert string[l:m].islower()
        if string[l] == '"':
            assert string[m-1] == '"'
            pred = Pred.from_string(string[l+1:m-1])
        else:
            pred = Pred.from_string(string[l:m])
    l = find_next(string, start=r+1)
    r = find_previous(string, start=l, end=len(string)-1) + 1
    attributes = _read_attributes(string[l:r], not_lower=('carg',))
    label = _read_reference(attributes.pop('lbl'), vs, rs)
    carg = attributes.pop('carg', None)
    assert (carg is not None) == (isinstance(pred, GPred) and pred.name in ('basic_card', 'basic_numbered_hour', 'card', 'dofm', 'dofw', 'mofy', 'named', 'named_n', 'numbered_hour', 'ord', 'season', 'year_range', 'yofc')), (carg, pred)  # gpreds with CARG

    args = {}
    quantifier = False
    if isinstance(pred, GPred):
        if pred.name in ('def_explicit_q', 'def_implicit_q', 'def_poss_q', 'every_q', 'free_relative_q', 'free_relative_ever_q', 'idiom_q_i', 'number_q', 'pronoun_q', 'proper_q', 'some_q', 'udef_q', 'which_q'):  # all quantifier gpreds according to core.smi
            quantifier = True
            handle = _read_reference(attributes.pop('rstr'), vs, rs)
            args['rstr'] = handle
            if 'body' in attributes:
                handle = _read_reference(attributes.pop('body'), vs, rs)
                args['body'] = handle
        elif pred.name in ('discourse', 'implicit_conj'):  # all conjunction gpreds with potential L/R-HNDL according to core.smi
            if 'arg1' in attributes:
                ref = _read_reference(attributes.pop('arg1'), vs, rs)
                args['arg1'] = ref
                ref = _read_reference(attributes.pop('arg2'), vs, rs)
                args['arg2'] = ref
            else:
                ref = _read_reference(attributes.pop('l-index'), vs, rs)
                args['l-index'] = ref
                ref = _read_reference(attributes.pop('r-index'), vs, rs)
                args['r-index'] = ref
                if 'l-hndl' in attributes:
                    assert pred.name == 'implicit_conj'
                    handle = _read_reference(attributes.pop('l-hndl'), vs, rs)
                    args['l-hndl'] = handle
                    handle = _read_reference(attributes.pop('r-hndl'), vs, rs)
                    args['r-hndl'] = handle
        elif pred.name in ('fw_seq', 'num_seq'):  # all conjunction gpreds without potential L/R-HNDL according to core.smi
            ref = _read_reference(attributes.pop('l-index'), vs, rs)
            args['l-index'] = ref
            ref = _read_reference(attributes.pop('r-index'), vs, rs)
            args['r-index'] = ref
        elif pred.name == 'unknown':  # only predicate with ARG
            ref = _read_reference(attributes.pop('arg'), vs, rs)
            args['arg'] = ref
        else:  # regular gpreds
            for n in range(1, 4):
                role = 'arg' + str(n)
                if role not in attributes:
                    break
                ref = _read_reference(attributes.pop(role), vs, rs)
                args[role] = ref

    elif isinstance(pred, RealPred):
        assert pred.pos in 'acnpquvx'
        if pred.pos == 'q':  # quantifier
            quantifier = True
            handle = _read_reference(attributes.pop('rstr'), vs, rs)
            args['rstr'] = handle
            if 'body' in attributes:
                handle = _read_reference(attributes.pop('body'), vs, rs)
                args['body'] = handle
        elif pred.pos == 'c' and pred.lemma not in ('vice+versa',):  # conjunction
            if 'arg1' in attributes:
                ref = _read_reference(attributes.pop('arg1'), vs, rs)
                args['arg1'] = ref
                if 'arg2' in attributes:
                    ref = _read_reference(attributes.pop('arg2'), vs, rs)
                    args['arg2'] = ref
            else:
                ref = _read_reference(attributes.pop('l-index'), vs, rs)
                args['l-index'] = ref
                ref = _read_reference(attributes.pop('r-index'), vs, rs)
                args['r-index'] = ref
                if 'l-hndl' in attributes:  # optional L/R-HNDL
                    handle = _read_reference(attributes.pop('l-hndl'), vs, rs)
                    args['l-hndl'] = handle
                    handle = _read_reference(attributes.pop('r-hndl'), vs, rs)
                    args['r-hndl'] = handle
        else:  # regular realpreds
            for n in range(1, 5):
                role = 'arg' + str(n)
                if role not in attributes:
                    break
                ref = _read_reference(attributes.pop(role), vs, rs)
                args[role] = ref
    else:
        assert False

    if quantifier:
        intrinsic = _read_reference(attributes.pop('arg0'), vs, rs)
        var = None
    else:
        intrinsic, var = _read_variable(attributes.pop('arg0'), vs, rs)
    assert not attributes, 'Invalid attributes for predicate {}: {}'.format(pred, attributes)
    return ElemPred(label=label, pred=pred, intrinsic=intrinsic, carg=carg, args=args, cfrom=cfrom, cto=cto), var


def read_mrs(string):
    mrs = Mrs()
    icon_labels = {}
    assert string[0] == '[' and string[-1] == ']'
    assert 'RELS:' in string and 'HCONS:' in string and string.index('RELS:') < string.index('HCONS:') and ('ICONS:' not in string or string.index('HCONS:') < string.index('ICONS:'))
    # var check
    rs = _get_reference_replacements(string)
    vs = {}
    l = find_next(string, 1)
    r = string.index('RELS:', l)
    while string[r-1] == ' ':
        r -= 1
    attributes = _read_attributes(string[l:r])
    if 'top' in attributes:
        mrs.top_handle = _read_reference(attributes.pop('top'), vs, rs)
        assert mrs.top_handle.is_handle()
    elif 'ltop' in attributes:
        mrs.top_handle = _read_reference(attributes.pop('ltop'), vs, rs)
        assert mrs.top_handle.is_handle()
    if 'index' in attributes:
        mrs.index_ref = _read_reference(attributes.pop('index'), vs, rs)
        assert mrs.index_ref.is_event()
    assert not attributes
    l = string.index('RELS:', r) + 6
    l = string.index('<', l) + 2
    r = find_substring(string, '>', l)
    while '[' in string[l:r]:
        l = string.index('[', l)
        bracket = string.find('[', l + 1)
        m = string.index(']', l) + 1
        while bracket != -1 and bracket < m:
            bracket = string.find('[', m + 1)
            m = string.index(']', m + 1) + 1
        ep, var = _read_elempred(string[l:m], vs, rs)
        l = m
        if isinstance(ep.pred, GPred) and ep.pred.name[-2:] == '_d':
            assert ep.intrinsic.is_event() and var.sortinfo.sf == 'prop'
            # and len(var.sortinfo) == 2) or (var.sortinfo.sf == 'prop' and var.sortinfo.tense == 'untensed' and var.sortinfo.mood == 'indicative' and len(var.sortinfo) == 4))  # untensed the same as None
            assert len(ep.args) == 2
            assert ep.label not in icon_labels, 'ach gott'
            icon_labels[ep.label] = (ep.args['arg1'][1], ep.pred.name[:-2], ep.args['arg2'][1])  # remember label of icon node
        else:
            if var is not None:
                mrs.add_var(var)
            mrs.add_node(ep)
    # for old_ref, new_ref in rs.items():
    #     mrs.change_ref(old_ref, new_ref)
    l = r + 2

    l = string.index('HCONS:', l) + 7
    l = string.index('<', l) + 2
    r = string.index('>', l)
    values = string[l:r].split()
    assert len(values) % 3 == 0
    for hole, label in ((_read_reference(values[3*n], vs, rs), _read_reference(values[3*n+2], vs, rs)) for n in range(len(values) // 3)):
        if label in icon_labels:
            ref1, icon, ref2 = icon_labels.pop(label)
            mrs.add_icon(ref1, icon, ref2)
            mrs.add_hcon(hole, label, icon_label=True)
        else:
            mrs.add_hcon(hole, label)
    l = r + 1

    if 'ICONS:' in string[l:]:
        assert not icon_labels
        l = find_substring(string, '<', l) + 1
        assert l >= 0
        r = find_substring(string, '>', l)
        assert r >= 0
        m = find_substring(string, '[', l, r)
        while m >= 0:
            m2 = find_substring(string, ']', m+1)
            assert m2 >= 0
            string = string[:m] + string[m2+1:]
            r -= m2 - m + 1
            m = find_substring(string, '[', l, r)
        values = string[l:r].split()
        assert len(values) % 3 == 0
        for ref1, icon, ref2 in ((_read_reference(values[3*n], vs, rs), values[3*n+1], _read_reference(values[3*n+2], vs, rs)) for n in range(len(values) // 3)):
            mrs.add_icon(ref1, icon, ref2)
    # assert not vs, 'Invalid instantiated variables: {}'.format(vs)

    # set top !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if mrs.top_handle is not None:
        if mrs.top_handle in mrs.eps:
            mrs.top = mrs.head_ep(mrs.top_handle)
        elif mrs.top_handle in mrs.hcons:
            mrs.top = mrs.head_ep(mrs.hcons[mrs.top_handle])
        else:
            pass  # assert False, 'Top handle hole not specified: {}'.format(mrs)
    if mrs.index_ref is not None:
        if mrs.index_ref not in mrs.var_eps:
            if mrs.top is not None and str(mrs.top.pred) == 'subord':
                vs.pop(mrs.index_ref)
            else:
                assert mrs.index_ref in mrs.var_eps, 'Index reference non-existent: {}'.format(mrs)
        else:
            mrs.index = mrs.var_eps[mrs.index_ref]  # relationship between...
    # assert not vs
    assert mrs.valid()
    return mrs
