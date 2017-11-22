from shapeworld import util
from shapeworld.captions import Caption


class Proposition(Caption):

    __slots__ = ('proptype', 'clauses')

    def __init__(self, proptype, clauses):
        # implication, conditional ???
        assert proptype in ('conjunction', 'disjunction', 'exclusive-disjunction')
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

    def agreement(self, predication, world):
        if self.proptype == 'conjunction':
            return min(clause.agreement(predication=predication.get_sub_predication(), world=world) for clause in self.clauses)

        elif self.proptype == 'disjunction':
            return max(clause.agreement(predication=predication.get_sub_predication(), world=world) for clause in self.clauses)

        elif self.proptype == 'exclusive-disjunction':
            return float(sum(clause.agreement(predication=predication.get_sub_predication(), world=world) > 0.0 for clause in self.clauses) == 1) * 2.0 - 1.0
