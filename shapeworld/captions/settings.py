
class Settings(object):

    def __new__(cls):
        raise NotImplementedError

    min_distance = 0.15
    min_axis_distance = 0.1
    min_overlap = 0.1
    min_area = 0.005
    min_shade = 0.2
    min_quantifier = 0.005
