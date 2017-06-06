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
            assert connective in ('conjunction', 'disjunction', 'exclusive-disjunction')  # excl-disjunction, implication, conditional ???
            self.clauses = tuple(clauses)
            self.connective = connective

    def model(self):
        return {'component': 'proposition', 'clauses': [clause.model() for clause in self.clauses], 'connective': self.connective}

    def agreement(self, world):
        if not self.connective:
            return self.clauses[0].agreement(world)
        elif self.connective == 'conjunction':
            return min(clause.agreement(world) for clause in self.clauses)
        elif self.connective == 'disjunction':
            return max(clause.agreement(world) for clause in self.clauses)
        elif self.connective == 'exclusive-disjunction':
            return float(sum(clause.agreement(world) > 0.5 for clause in self.clauses) == 1)
