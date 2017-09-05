

class Clause(object):

    def __str__(self):
        return self.__class__.__name__

    def model(self):
        raise NotImplementedError

    def agreement(self, entities):  # returns 0.0, 0.5, 1.0
        raise NotImplementedError
