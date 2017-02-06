from collections import namedtuple
from math import cos, pi, sqrt
from random import choice, uniform
from shapeworld.point import Point


golden_ratio = sqrt(5.0) / 2.0 - 0.5
sqrt34 = sqrt(0.75)
cos18 = cos(0.1 * pi)
cos45 = sqrt(2.0) / 2.0


SquareShapeTuple = namedtuple('SquareShapeTuple', ('size_',))
RectangleShapeTuple = namedtuple('RectangleShapeTuple', ('extent_',))
TriangleShapeTuple = namedtuple('TriangleShapeTuple', ('extent_',))
PentagonShapeTuple = namedtuple('PentagonShapeTuple', ('extent_',))
CrossShapeTuple = namedtuple('CrossShapeTuple', ('size_',))
CircleShapeTuple = namedtuple('CircleShapeTuple', ('size_',))
SemicircleShapeTuple = namedtuple('SemicircleShapeTuple', ('size_',))
EllipseShapeTuple = namedtuple('EllipseShapeTuple', ('extent_',))


class Shape(object):
    __slots__ = ()

    def __eq__(self, other):
        if type(other) is str:
            return str(self) == other
        return self == other

    def __str__(self):
        raise NotImplementedError

    def __contains__(self, offset):
        raise NotImplementedError

    def distance(self, offset):
        raise NotImplementedError

    def extent(self):
        raise NotImplementedError

    def polygon(self):
        return None

    def area(self):
        raise NotImplementedError

    @staticmethod
    def random_instance(shapes, size_range, distortion_range):
        return choice(shapes).random_instance(size_range=size_range, distortion_range=distortion_range)


class WorldShape(Shape):
    __slots__ = ()

    def __str__(self):
        return 'world'

    def __contains__(self, offset):
        return abs(offset) < 0.5

    def distance(self, offset):
        return (abs(offset) - 0.5).positive().length()

    def extent(self):
        return Point(0.5, 0.5)

    def polygon(self):
        return (Point(-0.5, -0.5),
                Point(0.5, -0.5),
                Point(-0.5, 0.5),
                Point(0.5, 0.5))

    def area(self):
        return 1.0


class SquareShape(Shape, SquareShapeTuple):
    __slots__ = ()

    def __new__(cls, size):
        assert isinstance(size, float) and size > 0.0
        return SquareShapeTuple.__new__(cls, size)

    def __str__(self):
        return 'square'

    def __contains__(self, offset):
        return abs(offset) <= self.size_

    def distance(self, offset):
        return (abs(offset) - self.size_).positive().length()

    def extent(self):
        return Point(self.size_, self.size_)

    def polygon(self):
        return (Point(-self.size_, -self.size_),
                Point(self.size_, -self.size_),
                Point(-self.size_, self.size_),
                Point(self.size_, self.size_))

    def area(self):
        return self.size_ * self.size_

    @staticmethod
    def random_instance(size_range, distortion_range):
        return SquareShape(size=uniform(*size_range))


class RectangleShape(Shape, RectangleShapeTuple):
    __slots__ = ()

    def __new__(cls, extent):
        assert isinstance(extent, Point) and extent > 0.0
        return RectangleShapeTuple.__new__(cls, extent)

    def __str__(self):
        return 'rectangle'

    def __contains__(self, offset):
        return abs(offset) <= self.extent_

    def distance(self, offset):
        return (abs(offset) - self.extent_).positive().length()

    def extent(self):
        return self.extent_

    def polygon(self):
        return (Point(-self.extent_.x, -self.extent_.y),
                Point(self.extent_.x, -self.extent_.y),
                Point(-self.extent_.x, self.extent_.y),
                self.extent_)

    def area(self):
        return self.extent_.x * self.extent_.y

    @staticmethod
    def random_instance(size_range, distortion_range):
        size = uniform(*size_range)
        distortion = uniform(*distortion_range)
        return RectangleShape(extent=Point(size, size / distortion))


class TriangleShape(Shape, TriangleShapeTuple):
    __slots__ = ()

    def __new__(cls, size):
        assert isinstance(size, float) and size > 0.0
        return TriangleShapeTuple.__new__(cls, Point(size, size * sqrt34))

    def __str__(self):
        return 'triangle'

    def __contains__(self, offset):
        return offset.y >= -self.extent_.y and 2.0 * abs(offset.x) / self.extent_.x + offset.y / self.extent_.y <= 1.0

    def distance(self, offset):
        if offset.y < -self.extent_.y:
            return (abs(offset) - self.extent_).positive().length()
        else:
            offset = Point(abs(offset.x), offset.y + self.extent_.y)
            linear = min(max(offset.y - offset.x + self.extent_.x, 0.0) / (self.extent_.x + 2.0 * self.extent_.y), 1.0)
            return Point(offset.x - (1.0 - linear) * self.extent_.x, offset.y - linear * 2.0 * self.extent_.y).positive().length()

    def extent(self):
        return self.extent_

    def polygon(self):
        return (Point(-self.extent_.x, -self.extent_.y),
                Point(self.extent_.x, -self.extent_.y),
                Point(0.0, self.extent_.y))

    def area(self):
        return self.extent_.x * self.extent_.y * 0.5

    @staticmethod
    def random_instance(size_range, distortion_range):
        return TriangleShape(size=uniform(*size_range))  # for equilateral


class PentagonShape(Shape, PentagonShapeTuple):
    __slots__ = ()

    def __new__(cls, size):
        assert isinstance(size, float) and size > 0.0
        return PentagonShapeTuple.__new__(cls, Point(size, size * cos18))

    def __str__(self):
        return 'pentagon'

    def __contains__(self, offset):
        return (offset.y >= -self.extent_.y and
                (offset.y + self.extent_.y) >= ((abs(offset.x) - golden_ratio * self.extent_.x) / ((1.0 - golden_ratio) * self.extent_.x) * (golden_ratio * 2.0 * self.extent_.y)) and
                (offset.y - (golden_ratio - 0.5) * 2.0 * self.extent_.y) <= ((1.0 - abs(offset.x) / self.extent_.x) * (1.0 - golden_ratio) * 2.0 * self.extent_.y))

    def distance(self, offset):
        offset = Point(abs(offset.x), offset.y + self.extent_.y - golden_ratio * 2.0 * self.extent_.y)
        if offset.y < 0.0:
            y_length = golden_ratio * 2.0 * self.extent_.y
            if offset.x < golden_ratio * self.extent_.x:
                return max(-offset.y - y_length, 0.0)
            else:
                offset = Point(offset.x - golden_ratio * self.extent_.x, -offset.y)
                x_length = (1.0 - golden_ratio) * self.extent_.x
                linear = min(max(offset.y - offset.x + x_length, 0.0) / (x_length + y_length), 1.0)
                return Point(offset.x - (1.0 - linear) * x_length, offset.y - linear * y_length).positive().length()
        else:
            y_length = (1.0 - golden_ratio) * 2.0 * self.extent_.y
            linear = min(max(offset.y - offset.x + self.extent_.x, 0.0) / (self.extent_.x + y_length), 1.0)
            return Point(offset.x - (1.0 - linear) * self.extent_.x, offset.y - linear * y_length).positive().length()

    def extent(self):
        return self.extent_

    def polygon(self):
        x_extent = golden_ratio * self.extent_.x
        y_extent = golden_ratio * 2.0 * self.extent_.y - self.extent_.y
        return (Point(-x_extent, -self.extent_.y),
                Point(x_extent, -self.extent_.y),
                Point(self.extent_.x, y_extent),
                Point(0.0, self.extent_.y),
                Point(-self.extent_.x, y_extent))

    def area(self):
        return (4.0 * golden_ratio * golden_ratio * self.extent_.x * self.extent_.y +
                2.0 * golden_ratio * (1.0 - golden_ratio) * self.extent_.x * self.extent_.y +
                2.0 * (1.0 - golden_ratio) * self.extent_.x * self.extent_.y)

    @staticmethod
    def random_instance(size_range, distortion_range):
        return PentagonShape(size=uniform(*size_range))  # for equilateral


class CrossShape(Shape, CrossShapeTuple):
    __slots__ = ()

    def __new__(cls, size):
        assert isinstance(size, float) and size > 0.0
        return CrossShapeTuple.__new__(cls, size)

    def __str__(self):
        return 'cross'

    def __contains__(self, offset):
        offset = abs(offset)
        return offset <= self.size_ and not (offset - Point(self.size_ / 3.0, self.size_ / 3.0)).positive() > 0.0

    def distance(self, offset):
        offset = abs(offset)
        if offset.x > offset.y:
            return (offset - Point(self.size_, self.size_ / 3.0)).positive().length()
        else:
            return (offset - Point(self.size_ / 3.0, self.size_)).positive().length()

    def extent(self):
        return Point(self.size_, self.size_)

    def polygon(self):
        return (Point(-self.size_, -self.size_ / 3.0),
                Point(-self.size_ / 3.0, -self.size_),
                Point(self.size_ / 3.0, -self.size_),
                Point(self.size_, -self.size_ / 3.0),
                Point(self.size_, self.size_ / 3.0),
                Point(self.size_ / 3.0, self.size_),
                Point(-self.size_ / 3.0, self.size_),
                Point(-self.size_, self.size_ / 3.0))

    def area(self):
        return self.size_ * self.size_ * 5.0 / 9.0

    @staticmethod
    def random_instance(size_range, distortion_range):
        return CrossShape(size=uniform(*size_range))


class CircleShape(Shape, CircleShapeTuple):
    __slots__ = ()

    def __new__(cls, size):
        assert isinstance(size, float) and size > 0.0
        return CircleShapeTuple.__new__(cls, size)

    def __str__(self):
        return 'circle'

    def __contains__(self, offset):
        return offset.length() <= self.size_

    def distance(self, offset):
        return max(offset.length() - self.size_, 0.0)

    def extent(self):
        return Point(self.size_, self.size_)

    def polygon(self):
        curve_extent = (1.0 - cos45) * self.size_
        return (Point(-self.size_, -curve_extent),
                Point(-curve_extent, -self.size_),
                Point(curve_extent, -self.size_),
                Point(self.size_, -curve_extent),
                Point(self.size_, curve_extent),
                Point(curve_extent, self.size_),
                Point(-curve_extent, self.size_),
                Point(-self.size_, curve_extent))

    def area(self):
        return pi * self.size_ * self.size_

    @staticmethod
    def random_instance(size_range, distortion_range):
        return CircleShape(size=uniform(*size_range))


class SemicircleShape(Shape, SemicircleShapeTuple):
    __slots__ = ()

    def __new__(cls, size):
        assert isinstance(size, float) and size > 0.0
        return SemicircleShapeTuple.__new__(cls, size)

    def __str__(self):
        return 'semicircle'

    def __contains__(self, offset):
        offset += Point(0.0, self.size_ / 2.0)
        return offset.length() <= self.size_ and offset.y >= 0.0

    def distance(self, offset):
        offset += Point(0.0, self.size_ / 2.0)
        if offset.y < 0.0:
            return (abs(offset) - Point(self.size_, 0.0)).positive().length()
        else:
            return max(offset.length() - self.size_, 0.0)

    def extent(self):
        return Point(self.size_, self.size_ / 2.0)

    def polygon(self):
        y_extent = self.size_ / 2.0
        curve_extent = (1.0 - cos45) * self.size_
        return (Point(-self.size_, -y_extent),
                Point(self.size_, -y_extent),
                Point(self.size_, -y_extent + curve_extent),
                Point(curve_extent, y_extent),
                Point(-curve_extent, y_extent),
                Point(-self.size_, -y_extent + curve_extent))

    def area(self):
        return pi * self.size_ * self.size_ / 2.0

    @staticmethod
    def random_instance(size_range, distortion_range):
        return SemicircleShape(size=uniform(*size_range))


class EllipseShape(Shape, EllipseShapeTuple):
    __slots__ = ()

    def __new__(cls, extent):
        assert isinstance(extent, Point) and extent > 0.0
        return EllipseShapeTuple.__new__(cls, extent)

    def __str__(self):
        return 'ellipse'

    def __contains__(self, offset):
        return (offset / self.extent_).length() <= 1.0

    def distance(self, offset):
        direction = offset / self.extent_
        direction_length = direction.length()
        if direction_length <= 1.0:
            return 0.0
        return ((direction - direction / direction_length) * self.extent_).length()

    def extent(self):
        return self.extent_

    def polygon(self):
        curve_extent = (1.0 - cos45) * self.extent_
        return (Point(-self.extent_.x, -curve_extent.y),
                Point(-curve_extent.x, -self.extent_.y),
                Point(curve_extent.x, -self.extent_.y),
                Point(self.extent_.x, -curve_extent.y),
                Point(self.extent_.x, curve_extent.y),
                Point(curve_extent.x, self.extent_.y),
                Point(-curve_extent.x, self.extent_.y),
                Point(-self.extent_.x, curve_extent.y))

    def area(self):
        return pi * self.extent_[0] * self.extent_[1]

    @staticmethod
    def random_instance(size_range, distortion_range):
        size = uniform(*size_range)
        distortion = uniform(*distortion_range)
        return EllipseShape(extent=Point(size, size / distortion))


Shape.shapes = {
    'square': SquareShape,
    'rectangle': RectangleShape,
    'triangle': TriangleShape,
    'pentagon': PentagonShape,
    'cross': CrossShape,
    'circle': CircleShape,
    'semicircle': SemicircleShape,
    'ellipse': EllipseShape}
