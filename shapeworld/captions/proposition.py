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

    def polish_notation(self, reverse=False):
        if reverse:
            return [rpn_symbol for clause in self.clauses for rpn_symbol in clause.polish_notation(reverse=reverse)] + \
                ['{}-{}{}'.format(self, self.proptype, len(self.clauses))]

        else:
            return ['{}-{}{}'.format(self, self.proptype, len(self.clauses))] + \
                [rpn_symbol for clause in self.clauses for rpn_symbol in clause.polish_notation(reverse=reverse)]

    def apply_to_predication(self, predication):
        assert predication.empty()
        predications = [predication]
        next_predication = predication
        for _ in range(len(self.clauses) - 1):
            next_predication = next_predication.sub_predication()
            predications.append(next_predication)
        for n, (clause, predication) in enumerate(reversed(list(zip(self.clauses, predications)))):
            clause.apply_to_predication(predication=predication)
            if n > 0:
                predication.sub_predications.append(predication.sub_predications.pop(0))

    def agreement(self, predication, world):
        if self.proptype == 'conjunction':
            agreement = self.clauses[0].agreement(predication=predication, world=world)
            next_predication = predication
            for clause in self.clauses[1:]:
                next_predication = next_predication.get_sub_predication(-1)
                agreement = min(agreement, clause.agreement(predication=next_predication, world=world))
            return agreement

        elif self.proptype == 'disjunction':
            agreement = self.clauses[0].agreement(predication=predication, world=world)
            next_predication = predication
            for clause in self.clauses[1:]:
                next_predication = next_predication.get_sub_predication(-1)
                agreement = max(agreement, clause.agreement(predication=next_predication, world=world))
            return agreement

        elif self.proptype == 'exclusive-disjunction':
            num_positive = int(self.clauses[0].agreement(predication=predication, world=world) > 0.0)
            next_predication = predication
            for clause in self.clauses[1:]:
                next_predication = next_predication.get_sub_predication(-1)
                num_positive += int(clause.agreement(predication=next_predication, world=world) > 0.0)
            return float(num_positive == 1) * 2.0 - 1.0

        elif self.proptype == 'implication':
            assert len(self.clauses) == 2
            # 1 => 0
            next_predication = predication.get_sub_predication(-1)
            return max(self.clauses[0].agreement(predication=predication, world=world), -self.clauses[1].agreement(predication=next_predication, world=world))

        elif self.proptype == 'equivalence':
            pos_agreement = self.clauses[0].agreement(predication=predication, world=world)
            neg_agreement = -pos_agreement
            next_predication = predication
            for clause in self.clauses[1:]:
                next_predication = next_predication.get_sub_predication(-1)
                agreement = clause.agreement(predication=next_predication, world=world)
                pos_agreement = min(pos_agreement, agreement)
                neg_agreement = min(neg_agreement, -agreement)
            return max(pos_agreement, neg_agreement)

        else:
            assert False
