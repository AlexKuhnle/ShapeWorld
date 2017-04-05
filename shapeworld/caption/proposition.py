from shapeworld.caption import Clause, Relation, Quantifier


class Proposition(Clause):

    __slots__ = ('clauses', 'connective')

    def __init__(self, clauses, connective=None):
        if connective is None:
            assert isinstance(clauses, Clause)
            self.clauses = (clauses,)
            self.connective = None
        else:
            assert isinstance(clauses, tuple) or isinstance(clauses, list)
            assert all(isinstance(clause, Clause) for clause in clauses)
            assert connective in ('conjunction', 'disjunction')  # excl-disjunction, implication, conditional ???
            self.clauses = tuple(clauses)
            self.connective = connective

    def agreement(self, world):
        if not self.connective:
            return self.clauses[0].agreement(world)
        elif self.connective == 'and':
            return min(clause.agreement(world) for clause in self.clauses)
        elif self.connective == 'or':
            return max(clause.agreement(world) for clause in self.clauses)
