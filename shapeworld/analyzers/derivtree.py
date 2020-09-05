
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



class DerivTree:

    def __init__(self, parent, index, name):
        self.parent = parent
        self.index = index
        self.name = name

    def __str__(self):
        return '({}:{} {})'.format(self.index, self.name, ' '.join(str(child) for child in self))

    def __iter__(self):
        return iter(self.children)

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(iter(self.children))

    @property
    def isroot(self):
        return not self.parent

    @property
    def isleaf(self):
        raise NotImplementedError

    @property
    def children(self):
        raise NotImplementedError

    @property
    def leaves(self):
        raise NotImplementedError

    def get_leaf(self, token):
        for leaf in self.leaves:
            if leaf.name == token:
                return leaf
        return None

    def collapsed_isroot(self):
        return not self.collapsed_parent()

    def collapsed_parent(self):
        p = self.parent
        while len(p) == 1:
            p = p.parent
        return p

    def collapsed_children(self):
        cc = []
        for child in self.children:
            while len(child) == 1:
                child = child[0]
            cc.append(child)
        return cc

    def collapsed_str(self):
        return '({}:{} {})'.format(self.index, self.name, ' '.join(child.collapsed_str() for child in self.collapsed_children()))

    def _distance_up(self, token):
        if self.collapsed_isroot():
            return -1
        distance = self.collapsed_parent()._distance_down(token)
        if distance >= 0:
            return distance
        else:
            distance = self.collapsed_parent()._distance_up(token)
            return distance + 2

    def _distance_down(self, token):
        if self.name == token:
            return 0
        for child in self.collapsed_children():
            distance = child._distance_down(token)
            if distance >= 0:
                return distance + 2
        return -1

    def collapsed_distance(self, token):
        return self._distance_up(token)

    @staticmethod
    def from_string(string, parent=None, start=None, end=None):
        l = 0 if start is None else start
        r = len(string) if end is None else end
        assert 0 <= l < r <= len(string)
        assert string[l] == '(' and string[r-1] == ')'
        l += 1
        r -= 1
        m = find_substring(string, '(', start=l, end=r, allow_escape=True)
        if m > string.find('[', l, r) != -1:
            m = -1
        if m >= 0:
            ss = string[l:m].split()
            if len(ss) == 1:
                assert parent is None
                rule_name = ss[0]
                assert rule_name[:5] == 'root_'
                tree = Node(parent, 0, rule_name)
            else:
                assert len(ss) == 5
                index = int(ss[0])
                # assert index > 0 or (index == 0 and parent is None)     doesn't hold
                rule_name = ss[1]
                x1 = float(ss[2])
                x2 = int(ss[3])
                x3 = int(ss[4])
                # other values?
                tree = Node(parent, index, rule_name)
            l = m
            while l >= 0:
                assert string[l] == '('
                m = find_substring(string, ')', start=l+1, end=r, allow_escape=True)
                assert m >= 0
                tree.add_child(DerivTree.from_string(string, parent=tree, start=l, end=m+1))
                l = find_next(string, start=m+1, end=r)
            return tree
        else:
            l = find_next(string, start=l, end=r)
            assert l >= 0
            assert string[l] == '"'
            m = string.index('"', l+1, r)
            assert m >= 0
            token = string[l+1:m]
            # assert token.isalpha() or (token[0] in '(“' and token[1:].isalpha()) or (token[-1] in '.,;-”' and token[:-1].isalpha()) or (' ' in token and token[:token.index(' ')].isalpha() and token[token.index(' ')+1:].isalpha()), token  # !!!!!!
            l = m + 1
            index = []
            phrase_structure = []
            while l >= 0:
                m = string.index('"token', l, r)
                index.append(int(string[l:m].strip()))
                l = find_next(string, start=m+6, end=r)
                assert l >= 0 and string[l] == '['
                m = find_substring(string, ']', start=l+1, end=r)
                assert m >= 0
                phrase_structure.append(string[l:m+1])
                l = find_next(string, start=m+1, end=r)
                assert string[l] == '"'
                l = find_next(string, start=l+1, end=r)
            # rest? phrase structure????????
            return Leaf(parent, index, token, phrase_structure)


class Node(DerivTree):

    def __init__(self, parent, index, rule_name):
        super(Node, self).__init__(parent, index, rule_name)
        self.subtrees = []

    def __len__(self):
        return len(self.subtrees)

    @property
    def isleaf(self):
        return False

    @property
    def children(self):
        return self.subtrees

    @property
    def leaves(self):
        for child in self.children:
            for leaf in child.leaves:
                yield leaf

    def add_child(self, tree):
        self.subtrees.append(tree)


class Leaf(DerivTree):

    def __init__(self, parent, index, token, phrase_structure):
        super(Leaf, self).__init__(parent, index, token)
        self.phrase_structure = phrase_structure

    def __len__(self):
        return 0

    @property
    def isleaf(self):
        return True

    @property
    def children(self):
        return []

    @property
    def leaves(self):
        yield self
