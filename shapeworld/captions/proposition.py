from shapeworld.captions import Caption


class Proposition(Caption):

    __slots__ = ('proptype', 'clauses')

    def __init__(self, proptype, clauses):
        assert proptype in ('conjunction', 'disjunction')
        assert len(clauses) >= 1 and all(isinstance(clause, Caption) for clause in clauses)
        self.proptype = proptype
        self.clauses = list(clauses)

    def model(self):
        return dict(
            component=str(self),
            proptype=self.proptype,
            clauses=[clause.model() for clause in self.clauses]
        )

    def reverse_polish_notation(self):
        return [rpn_symbol for clause in self.clauses for rpn_symbol in clause.reverse_polish_notation()] + \
            [str(len(self.clauses)), '{}-{}'.format(self, self.proptype)]

    def apply_to_predication(self, predication):
        for clause in self.clauses:
            clause_predication = predication.sub_predication()
            clause.apply_to_predication(predication=clause_predication)

    def agreement(self, predication, world):
        if self.proptype == 'conjunction':
            return min(clause.agreement(predication=predication.get_sub_predication(n), world=world) for n, clause in enumerate(self.clauses))

        elif self.proptype == 'disjunction':
            return max(clause.agreement(predication=predication.get_sub_predication(n), world=world) for n, clause in enumerate(self.clauses))
