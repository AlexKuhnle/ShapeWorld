from collections import namedtuple
from math import ceil, floor, sqrt, trunc
from random import uniform


PointTuple = namedtuple('PointTuple', ('x', 'y'))


class Point(PointTuple):

    def __new__(cls, x, y):
        assert isinstance(x, float) or isinstance(x, int) or isinstance(x, bool) or isinstance(x, str)
        assert isinstance(y, float) or isinstance(y, int) or isinstance(y, bool) or isinstance(y, str)
        if isinstance(x, str):
            x = float(x)
        if isinstance(y, str):
            y = float(y)
        return PointTuple.__new__(cls, x, y)

    def __str__(self):
        return '({}/{})'.format(self.x, self.y)

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
            return Point(self.x / other.x, self.y / other.y)
        else:
            return Point(self.x / other, self.y / other)

    def __floordiv__(self, other):
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, bool) or isinstance(other, Point)
        if isinstance(other, Point):
            return Point(self.x // other.x, self.y // other.y)
        else:
            return Point(self.x // other, self.y // other)

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
        return Point(other / self.x, other / self.y)

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

    def length(self):
        return sqrt(self.x * self.x + self.y * self.y)

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
    def range(start, end=None, size=None, step=None):
        assert isinstance(start, Point)
        assert end is None or isinstance(end, Point)
        assert size is None or isinstance(size, Point)
        if end is None:
            end = ceil(start)
            start = Point(0, 0)
        else:
            start = floor(start)
            end = ceil(end)
        assert start <= end
        if size is None:
            for x in range(start.x, end.x + 1):
                for y in range(start.y, end.y + 1):
                    yield Point(x, y)
        else:
            for x in range(start.x, end.x + 1):
                for y in range(start.y, end.y + 1):
                    point = Point(x, y)
                    yield point, Point(x / size[0], y / size[1])

    @staticmethod
    def random_instance(topleft, bottomright):
        return Point(uniform(topleft.x, bottomright.x), uniform(topleft.y, bottomright.y))


Point.zero = Point(0.0, 0.0)
Point.one = Point(1.0, 1.0)
Point.half = Point(0.5, 0.5)

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
