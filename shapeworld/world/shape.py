from math import cos, pi, sqrt
from random import choice, uniform
from shapeworld.world.point import Point


golden_ratio = sqrt(5.0) / 2.0 - 0.5
sqrt34 = sqrt(0.75)
cos18 = cos(0.1 * pi)
cos45 = sqrt(2.0) / 2.0


class Shape(object):

    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, Point) and 0.0 < size < 1.0
        self.extent = size / 2.0

    def __str__(self):
        raise NotImplementedError

    def model(self):
        return {'name': str(self), 'extent': self.extent.model()}

    def copy(self):
        raise NotImplementedError

    def __contains__(self, offset):
        raise NotImplementedError

    def distance(self, offset):
        raise NotImplementedError

    def area(self):
        raise NotImplementedError

    def polygon(self):
        return (Point(-self.extent.x, -self.extent.y),
                Point(self.extent.x, -self.extent.y),
                Point(-self.extent.x, self.extent.y),
                Point(self.extent.x, self.extent.y))

    @staticmethod
    def random_instance(shapes, size_range, distortion_range):
        return choice([Shape.shapes[shape] for shape in shapes]).random_instance(size_range, distortion_range)


class WorldShape(Shape):
    __slots__ = ()

    def __init__(self):
        pass

    @property
    def extent(self):
        return Point(0.5, 0.5)

    def __contains__(self, offset):
        return abs(offset) < 0.5

    def distance(self, offset):
        return (abs(offset) - 0.5).positive().length()

    def area(self):
        return 1.0


class SquareShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, float)
        return super().__init__(Point(size, size))

    def __str__(self):
        return 'square'

    def copy(self):
        return SquareShape(size=(self.extent.x * 2.0))

    def __contains__(self, offset):
        return abs(offset) <= self.extent

    def distance(self, offset):
        return (abs(offset) - self.extent).positive().length()

    def area(self):
        return 4.0 * self.extent.x * self.extent.y

    def polygon(self):
        return (Point(-self.extent.x, -self.extent.y),
                Point(self.extent.x, -self.extent.y),
                Point(-self.extent.x, self.extent.y),
                self.extent)

    @staticmethod
    def random_instance(size_range, distortion_range):
        return SquareShape(uniform(*size_range))


class RectangleShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, Point)
        return super().__init__(size)

    def __str__(self):
        return 'rectangle'

    def copy(self):
        return RectangleShape(size=(self.extent * 2.0))

    def __contains__(self, offset):
        return abs(offset) <= self.extent

    def distance(self, offset):
        return (abs(offset) - self.extent).positive().length()

    def area(self):
        return 4.0 * self.extent.x * self.extent.y

    def polygon(self):
        return (Point(-self.extent.x, -self.extent.y),
                Point(self.extent.x, -self.extent.y),
                Point(-self.extent.x, self.extent.y),
                self.extent)

    @staticmethod
    def random_instance(size_range, distortion_range):
        size = uniform(*size_range)
        distortion = uniform(*distortion_range)
        return RectangleShape(Point(size, size / distortion))


class TriangleShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, float)
        return super().__init__(Point(size, size * sqrt34))

    def __str__(self):
        return 'triangle'

    def copy(self):
        return TriangleShape(size=(self.extent.x * 2.0))

    def __contains__(self, offset):
        return offset.y >= -self.extent.y and 2.0 * abs(offset.x) / self.extent.x + offset.y / self.extent.y <= 1.0

    def distance(self, offset):
        if offset.y < -self.extent.y:
            return (abs(offset) - self.extent).positive().length()
        else:
            offset = Point(abs(offset.x), offset.y + self.extent.y)
            linear = min(max(offset.y - offset.x + self.extent.x, 0.0) / (self.extent.x + 2.0 * self.extent.y), 1.0)
            return Point(offset.x - (1.0 - linear) * self.extent.x, offset.y - linear * 2.0 * self.extent.y).positive().length()

    def area(self):
        return 2.0 * self.extent.x * self.extent.y

    def polygon(self):
        return (Point(-self.extent.x, -self.extent.y),
                Point(self.extent.x, -self.extent.y),
                Point(0.0, self.extent.y))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return TriangleShape(uniform(*size_range))  # for equilateral


class PentagonShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, float)
        return super().__init__(Point(size, size * cos18))

    def __str__(self):
        return 'pentagon'

    def copy(self):
        return PentagonShape(size=(self.extent.x * 2.0))

    def __contains__(self, offset):
        return (offset.y >= -self.extent.y and
                (offset.y + self.extent.y) >= ((abs(offset.x) - golden_ratio * self.extent.x) / ((1.0 - golden_ratio) * self.extent.x) * (golden_ratio * 2.0 * self.extent.y)) and
                (offset.y - (golden_ratio - 0.5) * 2.0 * self.extent.y) <= ((1.0 - abs(offset.x) / self.extent.x) * (1.0 - golden_ratio) * 2.0 * self.extent.y))

    def distance(self, offset):
        offset = Point(abs(offset.x), offset.y + self.extent.y - golden_ratio * 2.0 * self.extent.y)
        if offset.y < 0.0:
            y_length = golden_ratio * 2.0 * self.extent.y
            if offset.x < golden_ratio * self.extent.x:
                return max(-offset.y - y_length, 0.0)
            else:
                offset = Point(offset.x - golden_ratio * self.extent.x, -offset.y)
                x_length = (1.0 - golden_ratio) * self.extent.x
                linear = min(max(offset.y - offset.x + x_length, 0.0) / (x_length + y_length), 1.0)
                return Point(offset.x - (1.0 - linear) * x_length, offset.y - linear * y_length).positive().length()
        else:
            y_length = (1.0 - golden_ratio) * 2.0 * self.extent.y
            linear = min(max(offset.y - offset.x + self.extent.x, 0.0) / (self.extent.x + y_length), 1.0)
            return Point(offset.x - (1.0 - linear) * self.extent.x, offset.y - linear * y_length).positive().length()

    def area(self):
        return (4.0 * golden_ratio * golden_ratio * self.extent.x * self.extent.y +
                2.0 * golden_ratio * (1.0 - golden_ratio) * self.extent.x * self.extent.y +
                2.0 * (1.0 - golden_ratio) * self.extent.x * self.extent.y)

    def polygon(self):
        x_extent = golden_ratio * self.extent.x
        y_extent = golden_ratio * 2.0 * self.extent.y - self.extent.y
        return (Point(-x_extent, -self.extent.y),
                Point(x_extent, -self.extent.y),
                Point(self.extent.x, y_extent),
                Point(0.0, self.extent.y),
                Point(-self.extent.x, y_extent))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return PentagonShape(uniform(*size_range))  # for equilateral


class CrossShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, float)
        return super().__init__(Point(size, size))

    def __str__(self):
        return 'cross'

    def copy(self):
        return CrossShape(size=(self.extent.x * 2.0))

    def __contains__(self, offset):
        offset = abs(offset)
        return offset <= self.extent and not (offset - self.extent.x / 3.0).positive() > 0.0

    def distance(self, offset):
        offset = abs(offset)
        if offset.x > offset.y:
            return (offset - Point(self.extent.x, self.extent.y / 3.0)).positive().length()
        else:
            return (offset - Point(self.extent.x / 3.0, self.extent.y)).positive().length()

    def area(self):
        return 20.0 * self.extent.x * self.extent.y / 9.0

    def polygon(self):
        return (Point(-self.extent.x, -self.extent.y / 3.0),
                Point(-self.extent.x / 3.0, -self.extent.y),
                Point(self.extent.x / 3.0, -self.extent.y),
                Point(self.extent.x, -self.extent.y / 3.0),
                Point(self.extent.x, self.extent.y / 3.0),
                Point(self.extent.x / 3.0, self.extent.y),
                Point(-self.extent.x / 3.0, self.extent.y),
                Point(-self.extent.x, self.extent.y / 3.0))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return CrossShape(uniform(*size_range))


class CircleShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, float)
        return super().__init__(Point(size, size))

    def __str__(self):
        return 'circle'

    def copy(self):
        return CircleShape(size=(self.extent.x * 2.0))

    def __contains__(self, offset):
        return offset.length() <= self.extent.x

    def distance(self, offset):
        return max(offset.length() - self.extent.x, 0.0)

    def area(self):
        return pi * self.extent.x * self.extent.y

    def polygon(self):
        curve_extent = (1.0 - cos45) * self.extent.x
        return (Point(-self.extent.x, -curve_extent),
                Point(-curve_extent, -self.extent.y),
                Point(curve_extent, -self.extent.y),
                Point(self.extent.x, -curve_extent),
                Point(self.extent.x, curve_extent),
                Point(curve_extent, self.extent.y),
                Point(-curve_extent, self.extent.y),
                Point(-self.extent.x, curve_extent))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return CircleShape(uniform(*size_range))


class SemicircleShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, float)
        return super().__init__(Point(size, size * 0.5))

    def __str__(self):
        return 'semicircle'

    def copy(self):
        return SemicircleShape(size=(self.extent.x * 2.0))

    def __contains__(self, offset):
        offset += Point(0.0, self.extent.y)
        return offset.length() <= self.extent.x and offset.y >= 0.0

    def distance(self, offset):
        offset += Point(0.0, self.extent.y)
        if offset.y < 0.0:
            return (abs(offset) - Point(self.extent.x, 0.0)).positive().length()
        else:
            return max(offset.length() - self.extent.x, 0.0)

    def area(self):
        return pi * self.extent.x * self.extent.y / 2.0

    def polygon(self):
        curve_extent = (1.0 - cos45) * self.extent.x
        return (Point(-self.extent.x, -self.extent.y),
                Point(self.extent.x, -self.extent.y),
                Point(self.extent.x, -self.extent.y + curve_extent),
                Point(curve_extent, self.extent.y),
                Point(-curve_extent, self.extent.y),
                Point(-self.extent.x, -self.extent.y + curve_extent))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return SemicircleShape(uniform(*size_range))


class EllipseShape(Shape):
    __slots__ = ('extent',)

    def __init__(self, size):
        assert isinstance(size, Point)
        return super().__init__(size)

    def __str__(self):
        return 'ellipse'

    def copy(self):
        return EllipseShape(size=(self.extent * 2.0))

    def __contains__(self, offset):
        return (offset / self.extent).length() <= 1.0

    def distance(self, offset):
        direction = offset / self.extent
        direction_length = direction.length()
        if direction_length <= 1.0:
            return 0.0
        return ((direction - direction / direction_length) * self.extent).length()

    def area(self):
        return pi * self.extent.x * self.extent.y

    def polygon(self):
        curve_extent = (1.0 - cos45) * self.extent
        return (Point(-self.extent.x, -curve_extent.y),
                Point(-curve_extent.x, -self.extent.y),
                Point(curve_extent.x, -self.extent.y),
                Point(self.extent.x, -curve_extent.y),
                Point(self.extent.x, curve_extent.y),
                Point(curve_extent.x, self.extent.y),
                Point(-curve_extent.x, self.extent.y),
                Point(-self.extent.x, curve_extent.y))

    @staticmethod
    def random_instance(size_range, distortion_range):
        size = uniform(*size_range)
        distortion = uniform(*distortion_range)
        return EllipseShape(Point(size, size / distortion))


Shape.shapes = {
    'square': SquareShape,
    'rectangle': RectangleShape,
    'triangle': TriangleShape,
    'pentagon': PentagonShape,
    'cross': CrossShape,
    'circle': CircleShape,
    'semicircle': SemicircleShape,
    'ellipse': EllipseShape
}
