from __future__ import division
from shapeworld.caption import Clause, Noun, Relation


class Existential(Clause):

    __slots__ = ('subject', 'verb')

    def __init__(self, subject, verb):
        assert isinstance(subject, Noun)
        assert isinstance(verb, Relation)
        self.subject = subject
        self.verb = verb

    def model(self):
        return {'component': 'existential', 'subject': self.subject.model(), 'verb': self.verb.model()}

    def agreement(self, entities):
        verb_entities = self.verb.agreeing_entities(entities=entities)
        if len(self.subject.agreeing_entities(entities=verb_entities)) > 0:
            return 1.0
        elif len(self.verb.disagreeing_entities(entities=entities)) == len(entities) or len(self.subject.disagreeing_entities(entities=verb_entities)) == len(entities):
            return 0.0
        else:
            return 0.5
