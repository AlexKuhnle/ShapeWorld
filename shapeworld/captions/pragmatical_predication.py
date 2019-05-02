from random import choice
from shapeworld import util
from shapeworld.captions import Predicate, EntityType


class PragmaticalPredication(object):

    def __init__(self, agreeing, ambiguous=None, disagreeing=None, sub_predications=None):
        self.agreeing = list(agreeing)
        self.ambiguous = list() if ambiguous is None else list(ambiguous)
        self.disagreeing = list() if disagreeing is None else list(disagreeing)
        self.sub_predications = list() if sub_predications is None else list(sub_predications)
        self.entities = sorted((self.agreeing + self.ambiguous + self.disagreeing), key=(lambda e: e.id))
        self.not_disagreeing = sorted((self.agreeing + self.ambiguous), key=(lambda e: e.id))

    def __str__(self):
        return '{{agreeing: {}, ambiguous: {}, disagreeing: {}}}'.format(len(self.agreeing), len(self.ambiguous), len(self.disagreeing))

    @property
    def num_entities(self):
        return len(self.entities)

    @property
    def num_agreeing(self):
        return len(self.agreeing)

    @property
    def num_not_disagreeing(self):
        return len(self.not_disagreeing)

    def copy(self, reset=False, exclude_sub_predications=False):
        if reset:
            return PragmaticalPredication(agreeing=self.entities)
        elif exclude_sub_predications:
            return PragmaticalPredication(agreeing=self.agreeing, ambiguous=self.ambiguous, disagreeing=self.disagreeing)
        else:
            return PragmaticalPredication(agreeing=self.agreeing, ambiguous=self.ambiguous, disagreeing=self.disagreeing, sub_predications=[predication.copy(exclude_sub_predications=False) for predication in self.sub_predications])

    def empty(self):
        return len(self.ambiguous) == 0 and len(self.disagreeing) == 0

    def implies(self, predicate, **kwargs):
        assert isinstance(predicate, Predicate)
        return util.all_and_any(predicate.pred_agreement(entity=entity, **kwargs) for entity in self.agreeing) and all(not predicate.pred_disagreement(entity=entity, **kwargs) for entity in self.ambiguous)

    def implied_by(self, predicate, **kwargs):
        assert isinstance(predicate, Predicate)
        return len(self.agreeing) > 0 and any(predicate.pred_agreement(entity=entity, **kwargs) for entity in self.entities) and util.all_and_any(predicate.pred_disagreement(entity=entity, **kwargs) for entity in self.disagreeing) and all(not predicate.pred_agreement(entity=entity, **kwargs) for entity in self.ambiguous)

    def tautological(self, predicate, **kwargs):
        assert isinstance(predicate, Predicate)
        return util.all_and_any(predicate.pred_agreement(entity=entity, **kwargs) for entity in self.agreeing) and \
            all(predicate.pred_disagreement(entity=entity, predication=self, **kwargs) for entity in self.disagreeing)

    def contradictory(self, predicate, **kwargs):
        assert isinstance(predicate, Predicate)
        return util.all_and_any(predicate.pred_disagreement(entity=entity, **kwargs) for entity in self.agreeing)

    def get_sub_predications(self):
        for predication in self.sub_predications:
            yield predication
            yield from predication.get_sub_predications()

    # def redundant_sub_predications(self):
    #     for m in range(len(self.sub_predications)):
    #         if self.equals(other=self.sub_predications[m]):
    #             return True
    #         # not recursive
    #         # if self.sub_predications[m].redundant_sub_predications():
    #         #     return True
    #         for n in range(m + 1, len(self.sub_predications)):
    #             if self.sub_predications[m].equals(other=self.sub_predications[n]):
    #                 return True
    #     return False

    def apply(self, predicate, **kwargs):
        assert isinstance(predicate, Predicate)
        assert not isinstance(predicate, EntityType)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Insert in sorted order ideally !!!
        for n in reversed(range(len(self.agreeing))):
            if predicate.pred_disagreement(entity=self.agreeing[n], **kwargs):
                entity = self.agreeing.pop(n)
                self.not_disagreeing.remove(entity)
                self.disagreeing.append(entity)
            elif not predicate.pred_agreement(entity=self.agreeing[n], **kwargs):
                entity = self.agreeing.pop(n)
                self.ambiguous.append(entity)
        for n in reversed(range(len(self.ambiguous))):
            if predicate.pred_disagreement(entity=self.ambiguous[n], **kwargs):
                entity = self.ambiguous.pop(n)
                self.not_disagreeing.remove(entity)
                self.disagreeing.append(entity)

        assert self.num_agreeing <= self.num_not_disagreeing <= self.num_entities
        assert self.agreeing == sorted(self.agreeing, key=(lambda e: e.id))

    def sub_predication(self, reset=False, predication=None):
        assert not reset or predication is None
        if predication is None:
            predication = self.copy(reset=reset)
        self.sub_predications.append(predication)
        return predication

    def get_sub_predication(self, index):
        if index < len(self.sub_predications):
            return self.sub_predications[index]
        else:
            return None

    def __eq__(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        return all(entity1 == entity2 for entity1, entity2 in zip(self.agreeing, other.agreeing)) and all(entity in other.ambiguous for entity in self.ambiguous) and all(entity in self.ambiguous for entity in other.ambiguous)  # since ambiguous not in order

    def __le__(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        return all(entity in other.agreeing for entity in self.agreeing) and all(entity in other.not_disagreeing for entity in self.ambiguous)

    def __ge__(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        return all(entity in self.agreeing for entity in other.agreeing) and all(entity in self.not_disagreeing for entity in other.ambiguous)

    def disjoint(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        return (len(self.agreeing) == 0 or len(other.agreeing) == 0) or all(entity not in other.not_disagreeing for entity in self.agreeing) and all(entity not in self.not_disagreeing for entity in other.agreeing)

    def equals(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        return (len(self.agreeing) > 0 or len(other.agreeing) > 0) and all(entity in other.not_disagreeing for entity in self.agreeing) and all(entity in self.not_disagreeing for entity in other.agreeing)

    def union(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        agreeing = list()
        ambiguous = list()
        disagreeing = list()
        for entity in self.entities:
            if entity in self.agreeing or entity in other.agreeing:
                agreeing.append(entity)
            elif entity in self.disagreeing and entity in other.disagreeing:
                disagreeing.append(entity)
            else:
                ambiguous.append(entity)
        return PragmaticalPredication(agreeing=agreeing, ambiguous=ambiguous, disagreeing=disagreeing)

    def intersect(self, other):
        assert all(entity1 == entity2 for entity1, entity2 in zip(self.entities, other.entities))
        agreeing = list()
        ambiguous = list()
        disagreeing = list()
        for entity in self.entities:
            if entity in self.agreeing and entity in other.agreeing:
                agreeing.append(entity)
            elif entity in self.disagreeing or entity in other.disagreeing:
                disagreeing.append(entity)
            else:
                ambiguous.append(entity)
        return PragmaticalPredication(agreeing=agreeing, ambiguous=ambiguous, disagreeing=disagreeing)
