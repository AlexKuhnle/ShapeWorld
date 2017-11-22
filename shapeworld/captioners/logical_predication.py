
class LogicalPredication(object):

    def __init__(self, predicates=None):
        self.predicates = set() if predicates is None else set(predicates)

    def copy(self, reset=False):
        if reset:
            return LogicalPredication()
        else:
            return LogicalPredication(predicates=self.predicates)

    def empty(self):
        return len(self.predicates) == 0

    def redundant(self, predicate):
        return predicate in self.predicates

    def tautological(self, predicates):
        return set(predicates) <= self.predicates

    def apply(self, predicate):
        assert isinstance(predicate, str)
        self.predicates.add(predicate)

    def equals(self, other):
        return self.predicates == other.predicates

    def union(self, other):
        return LogicalPredication(predicates=(self.predicates | other.predicates))

    def intersect(self, other):
        return LogicalPredication(predicates=(self.predicates & other.predicates))
