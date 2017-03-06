
class Caption(object):

    def agreeing_entities(self, entities):  # returns subset of entities
        raise NotImplementedError

    def agreement(self, world):  # returns [0, 1]
        raise NotImplementedError

    @staticmethod
    def none_agreeing_entities(entities):
        return None

    @staticmethod
    def all_agreeing_entities(entities):
        return entities

    @staticmethod
    def none_agreement(world):
        return None
