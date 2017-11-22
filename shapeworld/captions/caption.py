
class Caption(object):

    __slots__ = ()

    def __str__(self):
        return self.__class__.__name__

    def model(self):
        raise NotImplementedError

    def reverse_polish_notation(self):
        raise NotImplementedError

    def agreement(self, entities, world_entities=None):
        # returns -1.0, 0.0, 1.0
        raise NotImplementedError
