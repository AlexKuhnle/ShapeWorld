from __future__ import division
from collections import namedtuple
from math import ceil, cos, floor, pi, sin, sqrt, trunc
from operator import __truediv__
from random import uniform


PointTuple = namedtuple('PointTuple', ('x', 'y'))


class Point(PointTuple):

    def __new__(cls, x, y):
        assert isinstance(x, float) or isinstance(x, int) or isinstance(x, bool) or isinstance(x, str)
        assert isinstance(y, float) or isinstance(y, int) or isinstance(y, bool) or isinstance(y, str)
        if isinstance(x, str) and isinstance(y, str):
            x = float(x)
            y = float(y)
        return super(Point, cls).__new__(cls, x, y)

    @staticmethod
    def from_angle(angle):
        assert isinstance(angle, float) and 0.0 <= angle < 1.0
        angle = angle * 2.0 * pi
        return Point(cos(angle), sin(angle))

    def __str__(self):
        return '({}/{})'.format(self.x, self.y)

    def model(self):
        return {'x': self.x, 'y': self.y}

    @staticmethod
    def from_model(model):
        return Point(x=model['x'], y=model['y'])

    def lower(self):
        return min(self.x, self.y)

    def upper(self):
        return max(self.x, self.y)

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)

    def distance(self, other):
        x_diff = self.x - other.x
        y_diff = self.y - other.y
        return sqrt(x_diff * x_diff + y_diff * y_diff)

    def is_right(self, angle):
        assert isinstance(angle, float) and 0.0 <= angle < 1.0
        angle = angle * 2.0 * pi
        return self.x * sin(angle) - self.y * cos(angle) > 0.0

    def __eq__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        else:
            return self.x == other and self.y == other

    def __ne__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return self.x != other.x or self.y != other.y
        else:
            return self.x != other or self.y != other

    def __lt__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return self.x < other.x and self.y < other.y
        else:
            return self.x < other and self.y < other

    def __gt__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return self.x > other.x and self.y > other.y
        else:
            return self.x > other and self.y > other

    def __le__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return self.x <= other.x and self.y <= other.y
        else:
            return self.x <= other and self.y <= other

    def __ge__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return self.x >= other.x and self.y >= other.y
        else:
            return self.x >= other and self.y >= other

    def __pos__(self):
        return Point(self.x, self.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __add__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __sub__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __mul__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(__truediv__(self.x, other.x), __truediv__(self.y, other.y))
        else:
            return Point(__truediv__(self.x, other), __truediv__(self.y, other))

    def __floordiv__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x // other.x, self.y // other.y)
        else:
            return Point(self.x // other, self.y // other)

    def __div__(self, other):
        return self.__truediv__(other)

    def __mod__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x % other.x, self.y % other.y)
        else:
            return Point(self.x % other, self.y % other)

    def __divmod__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x // other.x, self.y // other.y), Point(self.x % other.x, self.y % other.y)
        else:
            return Point(self.x // other, self.y // other), Point(self.x % other, self.y % other)

    def __pow__(self, other, modulo=None):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if modulo is None:
            if isinstance(other, Point):
                return Point(self.x ** other.x, self.y ** other.y)
            else:
                return Point(self.x ** other, self.y ** other)
        else:
            if isinstance(other, Point):
                return Point(pow(self.x, other.x, modulo), pow(self.y, other.y, modulo))
            else:
                return Point(pow(self.x, other, modulo), pow(self.y, other, modulo))

    def __radd__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other + self.x, other + self.y)

    def __rsub__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other - self.x, other - self.y)

    def __rmul__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other * self.x, other * self.y)

    def __rtruediv__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(__truediv__(other, self.x), __truediv__(other, self.y))

    def __rfloordiv__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other // self.x, other // self.y)

    def __rmod__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other % self.x, other % self.y)

    def __rdivmod__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other // self.x, other // self.y), Point(other % self.x, other % self.y)

    def __rpow__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool)
        return Point(other ** self.x, other ** self.y)

    def __abs__(self):
        if isinstance(self.x, bool):
            return Point(self.x, self.y)
        else:
            return Point(abs(self.x), abs(self.y))

    def __round__(self, n=0):
        return Point(round(self.x, n), round(self.y, n))

    def __floor__(self):
        return Point(floor(self.x), floor(self.y))

    def __ceil__(self):
        return Point(ceil(self.x), ceil(self.y))

    def __trunc__(self):
        return Point(trunc(self.x), trunc(self.y))

    def square(self):
        return Point(self.x * self.x, self.y * self.y)

    def sum(self):
        return self.x + self.y

    def positive(self):
        if isinstance(self.x, bool):
            return Point(self.x, self.y)
        elif isinstance(self.x, int):
            return Point(max(self.x, 0), max(self.y, 0))
        else:
            return Point(max(self.x, 0.0), max(self.y, 0.0))

    def min(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(min(self.x, other.x), min(self.y, other.y))
        else:
            return Point(min(self.x, other), min(self.y, other))

    def max(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(max(self.x, other.x), max(self.y, other.y))
        else:
            return Point(max(self.x, other), max(self.y, other))

    def rotate(self, angle_sin, angle_cos):
        return Point(self.x * angle_cos - self.y * angle_sin, self.x * angle_sin + self.y * angle_cos)

    @staticmethod
    def range(start, end=None, size=None):
        assert isinstance(start, Point)
        assert end is None or isinstance(end, Point)
        assert size is None or isinstance(size, Point)
        if end is None:
            end = start.__ceil__()
            start = Point.izero
        else:
            start = start.__floor__()
            end = end.__ceil__()
        assert start <= end
        if size is None:
            for x in range(int(start.x), int(end.x)):
                for y in range(int(start.y), int(end.y)):
                    yield Point(x, y)
        else:
            size -= Point.ione
            for x in range(int(start.x), int(end.x)):
                for y in range(int(start.y), int(end.y)):
                    point = Point(x, y)
                    yield point, Point(x / size.x, y / size.y)

    @staticmethod
    def random_instance(topleft, bottomright):
        return Point(uniform(topleft.x, bottomright.x), uniform(topleft.y, bottomright.y))


Point.zero = Point(0.0, 0.0)
Point.one = Point(1.0, 1.0)
Point.neg_one = Point(-1.0, -1.0)
Point.half = Point(0.5, 0.5)

Point.izero = Point(0, 0)
Point.ione = Point(1, 1)
Point.right = Point(1, 0)
Point.top_right = Point(1, 1)
Point.top = Point(0, 1)
Point.top_left = Point(-1, 1)
Point.left = Point(-1, 0)
Point.bottom_left = Point(-1, -1)
Point.bottom = Point(0, -1)
Point.bottom_right = Point(1, -1)
Point.directions = (Point.right, Point.top, Point.left, Point.bottom)
Point.directions_ext = (Point.right, Point.top_right, Point.top, Point.top_left, Point.left, Point.bottom_left, Point.bottom, Point.bottom_right)
