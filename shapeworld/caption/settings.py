
class Settings(object):

    def __new__(cls):
        raise NotImplementedError

    min_quantifier = 0.1
    min_distance = 0.1
    min_overlap = 0.1
    min_area = 0.01
    min_shade = 0.2
