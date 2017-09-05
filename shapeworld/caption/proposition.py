from shapeworld.caption import Clause


class Proposition(Clause):

    __slots__ = ('proptype', 'clauses')

    def __init__(self, proptype, clauses):
        # excl-disjunction, implication, conditional ???
        assert proptype in ('conjunction', 'disjunction', 'exclusive-disjunction')
        assert len(clauses) >= 1 and all(isinstance(clause, Clause) for clause in clauses)
        self.proptype = proptype
        self.clauses = tuple(clauses)

    def model(self):
        return {'component': 'proposition', 'proptype': self.proptype, 'clauses': [clause.model() for clause in self.clauses]}

    def agreement(self, entities):
        if self.proptype == 'conjunction':
            return min(clause.agreement(entities) for clause in self.clauses)
        elif self.proptype == 'disjunction':
            return max(clause.agreement(entities) for clause in self.clauses)
        elif self.proptype == 'exclusive-disjunction':
            return float(sum(clause.agreement(entities) > 0.0 for clause in self.clauses) == 1)
