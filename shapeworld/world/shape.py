from __future__ import division
from math import cos, pi, sqrt
from random import choice, uniform
from shapeworld.util import quadratic_uniform
from shapeworld.world import Point


golden_ratio = sqrt(5.0) / 2.0 - 0.5
sqrt34 = sqrt(0.75)
cos18 = cos(0.1 * pi)
cos45 = sqrt(2.0) / 2.0

empirical_distortion_multiplier = 0.81  # for distortion_range = (2.0, 3.0)


class Shape(object):

    __slots__ = ('size',)

    def __init__(self, size):
        assert isinstance(size, Point) and 0.0 < size <= 1.0
        self.size = size / 2.0

    def __eq__(self, other):
        return (isinstance(other, Shape) and self.name == other.name) or (isinstance(other, str) and self.name == other)

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def model(self):
        return dict(name=self.name, size=self.size.model())

    @staticmethod
    def from_model(model):
        return Shape.shapes[model['name']](size=Point.from_model(model['size']))

    def copy(self):
        raise NotImplementedError

    def __contains__(self, offset):
        raise NotImplementedError

    def distance(self, offset):
        raise NotImplementedError

    def centrality(self, offset):
        raise NotImplementedError

    @property
    def area(self):
        raise NotImplementedError

    @staticmethod
    def relative_area():
        raise NotImplementedError

    def polygon(self):
        return (Point(-self.size.x, -self.size.y),
                Point(self.size.x, -self.size.y),
                Point(-self.size.x, self.size.y),
                Point(self.size.x, self.size.y))

    @staticmethod
    def get_shapes():
        return sorted(Shape.shapes.keys())

    @staticmethod
    def get_shape(name):
        return Shape.shapes[name]

    @staticmethod
    def random_instance(size_range, distortion_range, shape=None, shapes=None):
        if shape is not None:
            shape = Shape.get_shape(shape)
        elif shapes is not None:
            shape = choice([Shape.get_shape(shape) for shape in shapes])
        else:
            assert False
        return shape.random_instance(size_range, distortion_range)

    @staticmethod
    def test():
        for shape in Shape.get_shapes():
            shape_cls = Shape.get_shape(name=shape)
            shape_obj = shape_cls.random_instance(size_range=(1.0, 1.0), distortion_range=(2.0, 2.0))
            contains = 0
            for point in Point.range(start=Point(-50, -50), end=Point(50, 50)):
                if point / 100 in shape_obj:
                    contains += 1
            estimated_area = contains / 10000
            assert shape_obj.area <= 1.0, (shape, shape_obj.area)
            assert shape_obj.area == shape_cls.relative_area() or shape_obj.area == shape_cls.relative_area() / empirical_distortion_multiplier, (shape, shape_obj.area, shape_cls.relative_area())
            assert abs(shape_obj.area - estimated_area) < 0.011, (shape, abs(shape_obj.area - estimated_area))
        return True


class WorldShape(Shape):
    __slots__ = ()

    def __init__(self):
        pass

    @property
    def size(self):
        return Point(0.5, 0.5)

    def __contains__(self, offset):
        return abs(offset) < 0.5

    def distance(self, offset):
        return (abs(offset) - 0.5).positive().length()

    @property
    def area(self):
        return 1.0


class Square(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        if isinstance(size, float):
            size = Point(size, size)
        return super(Square, self).__init__(size=size)

    def copy(self):
        return Square(size=(self.size.x * 2.0))

    def __contains__(self, offset):
        return abs(offset) <= self.size

    def distance(self, offset):
        return (abs(offset) - self.size).positive().length()

    def centrality(self, offset):
        return max(((self.size - abs(offset)) / self.size).lower(), 0.0)

    @property
    def area(self):
        return 4.0 * self.size.x * self.size.y

    @staticmethod
    def relative_area():
        return 1.0

    def polygon(self):
        return (Point(-self.size.x, -self.size.y),
                Point(self.size.x, -self.size.y),
                Point(-self.size.x, self.size.y),
                self.size)

    @staticmethod
    def random_instance(size_range, distortion_range):
        return Square(quadratic_uniform(*size_range))


class Rectangle(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        return super(Rectangle, self).__init__(size)

    def copy(self):
        return Rectangle(size=(self.size * 2.0))

    def __contains__(self, offset):
        return abs(offset) <= self.size

    def distance(self, offset):
        return (abs(offset) - self.size).positive().length()

    def centrality(self, offset):
        return max(((self.size - abs(offset)) / self.size).lower(), 0.0)

    @property
    def area(self):
        return 4.0 * self.size.x * self.size.y

    @staticmethod
    def relative_area():
        return 0.5 * empirical_distortion_multiplier

    def polygon(self):
        return (Point(-self.size.x, -self.size.y),
                Point(self.size.x, -self.size.y),
                Point(-self.size.x, self.size.y),
                self.size)

    @staticmethod
    def random_instance(size_range, distortion_range):
        distortion = uniform(*distortion_range)
        if distortion_range[0] < distortion_range[1]:
            distortion_ratio = (distortion - distortion_range[0]) / (distortion_range[1] - distortion_range[0])
            min_size = size_range[0] + (size_range[1] - size_range[0]) * distortion_ratio * 0.5
            size = quadratic_uniform(min_size, size_range[1])
        else:
            size = quadratic_uniform(*size_range)
        return Rectangle(Point(size, size / distortion))


class Triangle(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        if isinstance(size, float):
            size = Point(size, size * sqrt34)
        return super(Triangle, self).__init__(size=size)

    def copy(self):
        return Triangle(size=(self.size.x * 2.0))

    def __contains__(self, offset):
        return offset.y >= -self.size.y and 2.0 * abs(offset.x) / self.size.x + offset.y / self.size.y <= 1.0

    def distance(self, offset):
        if offset.y < -self.size.y:
            return (abs(offset) - self.size).positive().length()
        else:
            offset = Point(abs(offset.x), offset.y + self.size.y)
            linear = min(max(offset.y - offset.x + self.size.x, 0.0) / (self.size.x + 2.0 * self.size.y), 1.0)
            size = Point((1.0 - linear) * self.size.x, linear * 2.0 * self.size.y)
            return (offset - size).positive().length()

    def centrality(self, offset):
        if offset.y < -self.size.y:
            return 0.0
        else:
            x = abs(offset.x) / ((1.0 - (offset.y + self.size.y) / (2.0 * self.size.y)) * self.size.x)
            if offset.y < -self.size.y / 3.0:
                offset = Point(abs(offset.x), -offset.y - self.size.y / 3.0)
                frac_x = max(offset.x / self.size.x, 2.0 / 3.0)
                # x = abs(offset.x) / ((1.0 - (offset.y + self.size.y) / (2.0 * self.size.y)) * self.size.x)
                y = min(offset.y / (frac_x * 2.0 * self.size.y / 3.0), 1.0)
                linear = offset.y / (self.size.y * 2.0 / 3.0)
                linear **= 2
                # linear = 1.0 - (1.0 - linear) ** 2
                # linear = 1.0
            else:
                offset = Point(abs(offset.x), offset.y + self.size.y / 3.0)
                frac_x = (offset.x / self.size.x) * 3.0 / 2.0
                assert 1.0 - frac_x >= offset.y / (4.0 * self.size.y / 3.0)
                y = offset.y / ((1.0 - frac_x) * 4.0 * self.size.y / 3.0)
                linear = offset.y / (self.size.y * 4.0 / 3.0)
                linear = 1.0 - (1.0 - linear) ** 2
            linear = 1.0 - (1.0 - linear) ** 2
            assert 0.0 <= x <= 1.0, x
            assert 0.0 <= y <= 1.0, y
            assert 0.0 <= linear <= 1.0, linear
            # return ((1.0 - y) + (1.0 - x)) / 2.0
            return linear * (1.0 - y) + (1.0 - linear) * (1.0 - x)

    @property
    def area(self):
        return 2.0 * self.size.x * self.size.y

    @staticmethod
    def relative_area():
        return 0.5 * sqrt34

    def polygon(self):
        return (Point(-self.size.x, -self.size.y),
                Point(self.size.x, -self.size.y),
                Point(0.0, self.size.y))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return Triangle(quadratic_uniform(*size_range))  # for equilateral


class Pentagon(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        if isinstance(size, float):
            size = Point(size, size * cos18)
        return super(Pentagon, self).__init__(size=size)

    def copy(self):
        return Pentagon(size=(self.size.x * 2.0))

    def __contains__(self, offset):
        return (offset.y >= -self.size.y and
                (offset.y + self.size.y) >= ((abs(offset.x) - golden_ratio * self.size.x) / ((1.0 - golden_ratio) * self.size.x) * (golden_ratio * 2.0 * self.size.y)) and
                (offset.y - (golden_ratio - 0.5) * 2.0 * self.size.y) <= ((1.0 - abs(offset.x) / self.size.x) * (1.0 - golden_ratio) * 2.0 * self.size.y))

    def distance(self, offset):
        offset = Point(abs(offset.x), offset.y + self.size.y - golden_ratio * 2.0 * self.size.y)
        if offset.y < 0.0:
            y_length = golden_ratio * 2.0 * self.size.y
            if offset.x < golden_ratio * self.size.x:
                return max(-offset.y - y_length, 0.0)
            else:
                offset = Point(offset.x - golden_ratio * self.size.x, -offset.y)
                x_length = (1.0 - golden_ratio) * self.size.x
                linear = min(max(offset.y - offset.x + x_length, 0.0) / (x_length + y_length), 1.0)
                size = Point((1.0 - linear) * x_length, linear * y_length)
                return (offset - size).positive().length()
        else:
            y_length = (1.0 - golden_ratio) * 2.0 * self.size.y
            linear = min(max(offset.y - offset.x + self.size.x, 0.0) / (self.size.x + y_length), 1.0)
            size = Point((1.0 - linear) * self.size.x, linear * y_length)
            return (offset - size).positive().length()

    def centrality(self, offset):
        # return 0.0
        offset = Point(abs(offset.x), offset.y + self.size.y - golden_ratio * 2.0 * self.size.y)
        if offset.y < 0.0:
            y_length = golden_ratio * 2.0 * self.size.y
            if offset.x < golden_ratio * self.size.x:
                return max(((self.size - abs(offset)) / self.size).lower(), 0.0)
                return max((y_length + offset.y) / y_length, 0.0)
            else:
                offset = Point(offset.x - golden_ratio * self.size.x, -offset.y)
                x_length = (1.0 - golden_ratio) * self.size.x
                linear = min(max(offset.y - offset.x + x_length, 0.0) / (x_length + y_length), 1.0)
                size = Point((1.0 - linear) * x_length, linear * y_length)
                return max(((size - offset) / size).lower(), 0.0)
        else:
            y_length = (1.0 - golden_ratio) * 2.0 * self.size.y
            linear = min(max(offset.y - offset.x + self.size.x, 0.0) / (self.size.x + y_length), 1.0)
            size = Point((1.0 - linear) * self.size.x, linear * y_length)
            return max(((size - offset) / size).lower(), 0.0)

        # return (abs(offset) - self.size).positive().length()
        # return max(((self.size - abs(offset)) / self.size).lower(), 0.0)

        # return max(offset.length() - self.size.x, 0.0)
        # return max((self.size.x - offset.length()) / self.size.x, 0.0)

    @property
    def area(self):
        return (4.0 * golden_ratio * golden_ratio * self.size.x * self.size.y +
                2.0 * golden_ratio * (1.0 - golden_ratio) * self.size.x * self.size.y +
                2.0 * (1.0 - golden_ratio) * self.size.x * self.size.y)

    @staticmethod
    def relative_area():
        return cos18 * (golden_ratio * golden_ratio + 0.5 * golden_ratio * (1.0 - golden_ratio) + 0.5 * (1.0 - golden_ratio))

    def polygon(self):
        x_size = golden_ratio * self.size.x
        y_size = golden_ratio * 2.0 * self.size.y - self.size.y
        return (Point(-x_size, -self.size.y),
                Point(x_size, -self.size.y),
                Point(self.size.x, y_size),
                Point(0.0, self.size.y),
                Point(-self.size.x, y_size))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return Pentagon(quadratic_uniform(*size_range))  # for equilateral


class Cross(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        if isinstance(size, float):
            size = Point(size, size)
        return super(Cross, self).__init__(size=size)

    def copy(self):
        return Cross(size=(self.size.x * 2.0))

    def __contains__(self, offset):
        offset = abs(offset)
        return offset <= self.size and not (offset - self.size.x / 3.0).positive() > 0.0

    def distance(self, offset):
        offset = abs(offset)
        if offset.x > offset.y:
            size = Point(self.size.x, self.size.y / 3.0)
        else:
            size = Point(self.size.x / 3.0, self.size.y)
        return (offset - size).positive().length()

    def centrality(self, offset):
        offset = abs(offset)
        if offset.x > self.size.x / 3.0:
            size = Point(self.size.x, self.size.y / 3.0)
        elif offset.y > self.size.y / 3.0:
            size = Point(self.size.x / 3.0, self.size.y)
        elif offset.x > offset.y:
            size = Point(self.size.x / 3.0 * 4.0, self.size.y / 3.0)
        else:
            size = Point(self.size.x / 3.0, self.size.y / 3.0 * 4.0)
        return max(((size - offset) / size).lower(), 0.0)

    @property
    def area(self):
        return 20.0 * self.size.x * self.size.y / 9.0

    @staticmethod
    def relative_area():
        return 5.0 / 9.0

    def polygon(self):
        return (Point(-self.size.x, -self.size.y / 3.0),
                Point(-self.size.x / 3.0, -self.size.y),
                Point(self.size.x / 3.0, -self.size.y),
                Point(self.size.x, -self.size.y / 3.0),
                Point(self.size.x, self.size.y / 3.0),
                Point(self.size.x / 3.0, self.size.y),
                Point(-self.size.x / 3.0, self.size.y),
                Point(-self.size.x, self.size.y / 3.0))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return Cross(quadratic_uniform(*size_range))


class Circle(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        if isinstance(size, float):
            size = Point(size, size)
        return super(Circle, self).__init__(size=size)

    def copy(self):
        return Circle(size=(self.size.x * 2.0))

    def __contains__(self, offset):
        return offset.length() <= self.size.x

    def distance(self, offset):
        return max(offset.length() - self.size.x, 0.0)

    def centrality(self, offset):
        return max((self.size.x - offset.length()) / self.size.x, 0.0)

    @property
    def area(self):
        return pi * self.size.x * self.size.y

    @staticmethod
    def relative_area():
        return 0.25 * pi

    def polygon(self):
        curve_size = (1.0 - cos45) * self.size.x
        return (Point(-self.size.x, -curve_size),
                Point(-curve_size, -self.size.y),
                Point(curve_size, -self.size.y),
                Point(self.size.x, -curve_size),
                Point(self.size.x, curve_size),
                Point(curve_size, self.size.y),
                Point(-curve_size, self.size.y),
                Point(-self.size.x, curve_size))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return Circle(quadratic_uniform(*size_range))


class Semicircle(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        if isinstance(size, float):
            size = Point(size, size * 0.5)
        return super(Semicircle, self).__init__(size=size)

    def copy(self):
        return Semicircle(size=(self.size.x * 2.0))

    def __contains__(self, offset):
        offset += Point(0.0, self.size.y)
        return offset.length() <= self.size.x and offset.y >= 0.0

    def distance(self, offset):
        offset += Point(0.0, self.size.y)
        if offset.y < 0.0:
            return (abs(offset) - Point(self.size.x, 0.0)).positive().length()
        else:
            return max(offset.length() - self.size.x, 0.0)

    def centrality(self, offset):
        offset += Point(0.0, self.size.y)
        if offset.y < 0.0:
            return 0.0
        else:
            return max((self.size.x - offset.length()) / self.size.x, 0.0)

    @property
    def area(self):
        return pi * self.size.x * self.size.y

    @staticmethod
    def relative_area():
        return 0.125 * pi

    def polygon(self):
        curve_size = (1.0 - cos45) * self.size.x
        return (Point(-self.size.x, -self.size.y),
                Point(self.size.x, -self.size.y),
                Point(self.size.x, -self.size.y + curve_size),
                Point(curve_size, self.size.y),
                Point(-curve_size, self.size.y),
                Point(-self.size.x, -self.size.y + curve_size))

    @staticmethod
    def random_instance(size_range, distortion_range):
        return Semicircle(quadratic_uniform(*size_range))


class Ellipse(Shape):
    __slots__ = ('size',)

    def __init__(self, size):
        return super(Ellipse, self).__init__(size)

    def copy(self):
        return Ellipse(size=(self.size * 2.0))

    def __contains__(self, offset):
        return (offset / self.size).length() <= 1.0

    def distance(self, offset):
        offset = abs(offset) / self.size
        offset -= offset / offset.length()
        return (offset * self.size).positive().length()

    def centrality(self, offset):
        offset = abs(offset) / self.size
        offset = offset / offset.length() - offset
        return max(offset.length(), 0.0)

    @property
    def area(self):
        return pi * self.size.x * self.size.y

    @staticmethod
    def relative_area():
        return 0.125 * pi * empirical_distortion_multiplier

    def polygon(self):
        curve_size = (1.0 - cos45) * self.size
        return (Point(-self.size.x, -curve_size.y),
                Point(-curve_size.x, -self.size.y),
                Point(curve_size.x, -self.size.y),
                Point(self.size.x, -curve_size.y),
                Point(self.size.x, curve_size.y),
                Point(curve_size.x, self.size.y),
                Point(-curve_size.x, self.size.y),
                Point(-self.size.x, curve_size.y))

    @staticmethod
    def random_instance(size_range, distortion_range):
        distortion = uniform(*distortion_range)
        if distortion_range[0] < distortion_range[1]:
            distortion_ratio = (distortion - distortion_range[0]) / (distortion_range[1] - distortion_range[0])
            min_size = size_range[0] + (size_range[1] - size_range[0]) * distortion_ratio * 0.5
            size = quadratic_uniform(min_size, size_range[1])
        else:
            size = quadratic_uniform(*size_range)
        return Ellipse(Point(size, size / distortion))


Shape.shapes = dict(
    square=Square,
    rectangle=Rectangle,
    triangle=Triangle,
    pentagon=Pentagon,
    cross=Cross,
    circle=Circle,
    semicircle=Semicircle,
    ellipse=Ellipse
)
