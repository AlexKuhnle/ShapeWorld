
class LogicalPredication(object):

    def __init__(self, predicates=None, blocked_preds=None):
        self.predicates = set() if predicates is None else set(predicates)
        self.blocked_preds = set() if blocked_preds is None else set(blocked_preds)

    def copy(self, reset=False):
        if reset:
            return LogicalPredication()
        else:
            return LogicalPredication(predicates=self.predicates, blocked_preds=self.blocked_preds)

    def empty(self):
        return len(self.predicates) == 0

    def redundant(self, predicate):
        return predicate in self.predicates

    def tautological(self, predicates):
        return set(predicates) <= self.predicates

    def blocked(self, predicate):
        return predicate in self.blocked_preds

    def apply(self, predicate):
        assert isinstance(predicate, str)
        self.predicates.add(predicate)

    def block(self, predicate):
        assert isinstance(predicate, str)
        assert predicate in self.predicates
        self.blocked_preds.add(predicate)

    def equals(self, other):
        return self.predicates == other.predicates

    def union(self, other):
        return LogicalPredication(predicates=(self.predicates | other.predicates))

    def intersect(self, other):
        return LogicalPredication(predicates=(self.predicates & other.predicates))
