from shapeworld.captions import Caption


class Proposition(Caption):

    __slots__ = ('proptype', 'clauses')

    def __init__(self, proptype, clauses):
        assert proptype in ('conjunction', 'disjunction', 'exclusive-disjunction', 'implication', 'equivalence')
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
            [str(len(self.clauses)), '{}-{}'.format(self, self.proptype)]  # two separate arguments, no tuple?

    def apply_to_predication(self, predication):
        for clause in self.clauses:
            clause_predication = predication.sub_predication()
            clause.apply_to_predication(predication=clause_predication)

    def agreement(self, predication, world):
        if self.proptype == 'conjunction':
            return min(clause.agreement(predication=predication.get_sub_predication(n), world=world) for n, clause in enumerate(self.clauses))

        elif self.proptype == 'disjunction':
            return max(clause.agreement(predication=predication.get_sub_predication(n), world=world) for n, clause in enumerate(self.clauses))

        elif self.proptype == 'exclusive-disjunction':
            return float(sum(clause.agreement(predication=predication.get_sub_predication(n), world=world) > 0.0 for n, clause in enumerate(self.clauses)) == 1) * 2.0 - 1.0

        elif self.proptype == 'implication':
            assert len(self.clauses) == 2
            # 1 => 0
            return max(self.clauses[0].agreement(predication=predication.get_sub_predication(0), world=world), -self.clauses[1].agreement(predication=predication.get_sub_predication(1), world=world))

        elif self.proptype == 'equivalence':
            return max(min(clause.agreement(predication=predication.get_sub_predication(n), world=world) for n, clause in enumerate(self.clauses)), min(-clause.agreement(predication=predication.get_sub_predication(n), world=world) for n, clause in enumerate(self.clauses)))

        else:
            assert False
