from __future__ import division
from math import cos, pi, sin
from random import choice, random
from shapeworld.util import Point
from shapeworld.world.shape import Shape
from shapeworld.world.color import Color
from shapeworld.world.texture import Texture


class Entity(object):

    __slots__ = ('id', 'shape', 'color', 'texture', 'center', 'rotation', 'rotation_sin', 'rotation_cos', 'topleft', 'bottomright')

    def __init__(self, shape, color, texture, center, rotation):
        assert isinstance(shape, Shape)
        assert isinstance(color, Color)
        assert isinstance(texture, Texture)
        assert isinstance(center, Point)
        assert isinstance(rotation, float) and 0.0 <= rotation < 1.0
        self.id = None
        self.shape = shape
        self.color = color
        self.texture = texture
        self.rotation = rotation
        self.rotation_sin = sin(-rotation * 2.0 * pi)
        self.rotation_cos = cos(-rotation * 2.0 * pi)
        self.set_center(center=center)

    def model(self):
        return {'id': self.id, 'shape': self.shape.model(), 'color': self.color.model(), 'texture': self.texture.model(), 'center': self.center.model(), 'rotation': self.rotation}

    def copy(self):
        return Entity(shape=self.shape.copy(), color=self.color.copy(), texture=self.texture.copy(), center=self.center, rotation=self.rotation)

    def rotate(self, offset):
        return offset.rotate(self.rotation_sin, self.rotation_cos)

    def __contains__(self, offset):
        return self.rotate(offset) in self.shape

    def distance(self, offset):
        return self.shape.distance(self.rotate(offset))

    def set_center(self, center):
        self.center = center
        inv_rot_sin = sin(self.rotation * 2.0 * pi)
        inv_rot_cos = cos(self.rotation * 2.0 * pi)
        topleft = Point.one
        bottomright = Point.zero
        for point in self.shape.polygon():
            point = point.rotate(inv_rot_sin, inv_rot_cos)
            topleft = topleft.min(point)
            bottomright = bottomright.max(point)
        self.topleft = topleft + center
        self.bottomright = bottomright + center

    def draw(self, world_array, world_size):
        shift = Point(2.0 / world_size.x, 2.0 / world_size.y)
        scale = 1.0 + 2.0 * shift
        topleft = (((self.topleft + 0.5 * shift) / scale) * world_size).max(Point.izero)
        bottomright = (((self.bottomright + 1.5 * shift) / scale) * world_size).min(world_size)
        color = self.color.get_color()
        for (x, y), point in Point.range(topleft, bottomright, world_size - Point.ione):
            point = point * scale - shift
            offset = point - self.center
            distance = self.distance(offset)
            if distance == 0.0:
                # assert offset in self
                world_array[y, x] = self.texture.get_color(color, offset)
            else:
                # assert offset not in self
                distance = max(1.0 - distance * min(*world_size), 0.0)
                world_array[y, x] = distance * self.texture.get_color(color, offset) + (1.0 - distance) * world_array[y, x]

    def overlaps(self, other):
        topleft1 = self.topleft
        bottomright1 = self.bottomright
        topleft2 = other.topleft
        bottomright2 = other.bottomright
        if bottomright1.x < topleft2.x or topleft1.x > bottomright2.x or bottomright1.y < topleft2.y or topleft1.y > bottomright2.y:
            return None
        else:
            return topleft1.max(topleft2), bottomright1.min(bottomright2)

    def collides(self, other, world_size, ratio=False, symmetric=True):
        overlap = self.overlaps(other)
        if not overlap:
            if ratio:
                return 0.0 if symmetric else (0.0, 0.0)
            else:
                return False
        topleft, bottomright = overlap
        topleft *= world_size
        bottomright *= world_size
        if ratio:
            granularity = 1.0 / world_size.x / world_size.y
            collision = 0.0
            for _, point in Point.range(topleft, bottomright, world_size - Point.ione):
                if ((point - self.center) in self) and ((point - other.center) in other):
                    collision += granularity
            if symmetric:
                return min(collision / self.shape.area(), collision / other.shape.area())
            else:
                return (collision / self.shape.area(), collision / other.shape.area())
        else:
            min_distance = 1.0 / world_size.x
            for _, point in Point.range(topleft, bottomright, world_size - Point.ione):
                if (self.distance(point - self.center) <= min_distance) and (other.distance(point - other.center) <= min_distance):
                    return True

    def not_overlaps(self, other):
        topleft1 = self.topleft
        bottomright1 = self.bottomright
        topleft2 = other.topleft
        bottomright2 = other.bottomright
        if topleft1.x < topleft2.x and topleft1.y < topleft2.y and bottomright1.x > bottomright2.x and bottomright1.y > bottomright2.y:
            return None
        elif (bottomright1 - topleft1).length() < (bottomright2 - topleft2).length():
            return topleft1, bottomright1
        else:
            return topleft2, bottomright2

    def not_collides(self, other, world_size, ratio=False, symmetric=True):
        not_overlap = self.not_overlaps(other)
        if not not_overlap:
            if ratio:
                return 0.0 if symmetric else (0.0, 0.0)
            else:
                return False
        topleft, bottomright = not_overlap
        topleft *= world_size
        bottomright *= world_size
        granularity = 1.0 / world_size.x / world_size.y
        collision = 0.0
        if ratio:
            for _, point in Point.range(topleft, bottomright, world_size - Point.ione):
                if ((point - self.center) not in self) and ((point - other.center) in other):
                    collision += granularity
            if symmetric:
                return min(collision / self.shape.area(), collision / other.shape.area())
            else:
                return (collision / self.shape.area(), collision / other.shape.area())
        else:
            for c, point in Point.range(topleft, bottomright, world_size - Point.ione):
                if (self.distance(point - self.center) > granularity) and (other.distance(point - other.center) <= granularity):
                    return True

    @staticmethod
    def random_instance(center, rotation, size_range, distortion_range, shade_range, shapes=None, colors=None, textures=None, combinations=None):
        # random color in texture
        assert (shapes and colors and textures) != bool(combinations)
        rotation = random() if rotation else 0.0
        if combinations:
            shape, color, texture = choice(combinations)
            shape = Shape.random_instance([shape], size_range, distortion_range)
            color = Color.random_instance([color], shade_range)
            texture = Texture.random_instance([texture], [c for c in Color.colors if c != color], shade_range)
        else:
            shape = Shape.random_instance(shapes, size_range, distortion_range)
            color = Color.random_instance(colors, shade_range)
            texture = Texture.random_instance(textures, [c for c in Color.colors if c != color], shade_range)
        return Entity(shape, color, texture, center, rotation)
