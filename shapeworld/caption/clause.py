

class Clause(object):

    def model(self):
        raise NotImplementedError

    def agreement(self, world):  # returns 0.0, 0.5, 1.0
        raise NotImplementedError
