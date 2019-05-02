
class Caption(object):

    __slots__ = ()

    def __str__(self):
        return self.__class__.__name__

    def model(self):
        raise NotImplementedError

    def polish_notation(self, reverse=False):
        raise NotImplementedError

    def apply_to_predication(self, predication):
        raise NotImplementedError

    def agreement(self, predication, world):
        # returns -1.0, 0.0, 1.0
        raise NotImplementedError
