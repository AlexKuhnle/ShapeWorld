from shapeworld.caption import Clause


class Proposition(Clause):

    __slots__ = ('proptype', 'clauses')

    def __init__(self, proptype, clauses):
        # excl-disjunction, implication, conditional ???
        assert proptype in ('modifier', 'noun', 'relation', 'existential', 'quantifier', 'conjunction', 'disjunction', 'exclusive-disjunction')
        if proptype in ('modifier', 'noun', 'relation', 'existential', 'quantifier'):
            assert isinstance(clauses, Clause)
            clauses = (clauses,)
        else:
            assert len(clauses) >= 1 and all(isinstance(clause, Clause) for clause in clauses)
        self.proptype = proptype
        self.clauses = tuple(clauses)

    def model(self):
        return {'component': 'proposition', 'proptype': self.proptype, 'clauses': [clause.model() for clause in self.clauses]}

    def agreement(self, entities):
        if self.proptype in ('modifier', 'noun', 'relation', 'existential', 'quantifier'):
            return self.clauses[0].agreement(entities)
        elif self.connective == 'conjunction':
            return min(clause.agreement(entities) for clause in self.clauses)
        elif self.connective == 'disjunction':
            return max(clause.agreement(entities) for clause in self.clauses)
        elif self.connective == 'exclusive-disjunction':
            return float(sum(clause.agreement(entities) > 0.5 for clause in self.clauses) == 1)
