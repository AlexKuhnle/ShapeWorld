from __future__ import division
from math import cos, pi, sin
from random import choice, random
from shapeworld.world import Point, Shape, Color, Texture


default_resolution = Point(100, 100)


class Entity(object):

    __slots__ = ('id', 'shape', 'color', 'texture', 'center', 'rotation', 'rotation_sin', 'rotation_cos', 'relative_topleft', 'relative_bottomright', 'topleft', 'bottomright', 'collisions')

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
        self.collisions = dict()

    def __hash__(self):
        assert self.id is not None
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.id is not None and other.id is not None and self.id == other.id

    def model(self):
        return dict(id=self.id, shape=self.shape.model(), color=self.color.model(), texture=self.texture.model(), center=self.center.model(), rotation=self.rotation, bounding_box=dict(topleft=self.topleft.model(), bottomright=self.bottomright.model()))

    @staticmethod
    def from_model(model):
        entity = Entity(shape=Shape.from_model(model['shape']), color=Color.from_model(model['color']), texture=Texture.from_model(model['texture']), center=Point.from_model(model['center']), rotation=model['rotation'])
        entity.id = model['id']
        return entity

    def __str__(self):
        return '({}, {}, {})'.format(self.shape.name, self.color.name, self.texture.name)

    def copy(self):
        return Entity(shape=self.shape.copy(), color=self.color.copy(), texture=self.texture.copy(), center=self.center, rotation=self.rotation)

    def rotate(self, offset):
        return offset.rotate(self.rotation_sin, self.rotation_cos)

    def __contains__(self, offset):
        return self.rotate(offset) in self.shape

    def distance(self, offset):
        return self.shape.distance(self.rotate(offset))

    def centrality(self, offset):
        return self.shape.centrality(self.rotate(offset))

    def set_center(self, center):
        self.center = center
        inv_rot_sin = sin(self.rotation * 2.0 * pi)
        inv_rot_cos = cos(self.rotation * 2.0 * pi)
        topleft = Point.one
        bottomright = Point.neg_one
        for point in self.shape.polygon():
            point = point.rotate(inv_rot_sin, inv_rot_cos)
            topleft = topleft.min(point)
            bottomright = bottomright.max(point)
        self.relative_topleft = topleft
        self.relative_bottomright = bottomright
        self.topleft = topleft + center
        self.bottomright = bottomright + center

    def draw(self, world_array, world_size, draw_fn=None):
        shift = Point(2.0 / world_size.x, 2.0 / world_size.y)
        scale = 1.0 + 2.0 * shift
        topleft = (((self.topleft) / scale) * world_size).max(Point.izero)  # + 0.5 * shift
        bottomright = (((self.bottomright + 2.0 * shift) / scale) * world_size).min(world_size)

        if draw_fn is None:
            color = self.color.get_color()
            world_length = min(*world_size)
            for (x, y), point in Point.range(topleft, bottomright, world_size):
                point = point * scale - shift
                offset = point - self.center
                distance = self.distance(offset)
                if distance == 0.0:
                    # assert offset in self
                    world_array[y, x] = self.texture.get_color(color, offset)
                else:
                    # assert offset not in self
                    distance = max(1.0 - distance * world_length, 0.0)
                    world_array[y, x] = distance * self.texture.get_color(color, offset) + (1.0 - distance) * world_array[y, x]
                # if distance == 0.0:
                #     centrality = self.centrality(offset)
                #     world_array[y, x] = (centrality, centrality, centrality) + (1.0 - centrality) * self.texture.get_color(color, offset)
                # else:
                #     assert self.centrality(offset) == 0.0

        else:
            for (x, y), point in Point.range(topleft, bottomright, world_size):
                point = point * scale - shift
                offset = point - self.center
                world_array[y, x] = draw_fn(value=world_array[y, x], entity=self, offset=offset)

        bounding_box = False
        if bounding_box:  # draw bounding box
            assert draw_fn is None
            x1 = world_size + 1
            x2 = -1
            y1 = world_size + 1
            y2 = -1
            for (x, y), point in Point.range(topleft, bottomright, world_size):
                if x < x1:
                    x1 = x
                if x > x2:
                    x2 = x
                if y < y1:
                    y1 = y
                if y > y2:
                    y2 = y
            for (x, y), point in Point.range(topleft, bottomright, world_size):
                if x == x1 or x == x2 or y == y1 or y == y2:
                    world_array[y, x] = color

    def overall_collision(self):
        return sum(self.collisions.values())

    def collides(self, other, ratio=False, symmetric=False, resolution=None):
        if other.id == self.id:
            if not ratio:
                return True
            elif symmetric:
                return 1.0
            else:
                return (1.0, 1.0)

        if other.id in self.collisions and self.id in other.collisions:
            if not ratio:
                return min(self.collisions[other.id], other.collisions[self.id]) > 0.0
            elif symmetric:
                return min(self.collisions[other.id], other.collisions[self.id])
            else:
                return (self.collisions[other.id], other.collisions[self.id])

        topleft1 = self.topleft
        bottomright1 = self.bottomright
        topleft2 = other.topleft
        bottomright2 = other.bottomright
        if bottomright1.x < topleft2.x or topleft1.x > bottomright2.x or bottomright1.y < topleft2.y or topleft1.y > bottomright2.y:
            if other.id is not None:
                self.collisions[other.id] = 0.0
            if self.id is not None:
                other.collisions[self.id] = 0.0
            if not ratio:
                return False
            elif symmetric:
                return 0.0
            else:
                return (0.0, 0.0)
        else:
            topleft, bottomright = topleft1.max(topleft2), bottomright1.min(bottomright2)

        if resolution is None:
            resolution = default_resolution
        topleft *= resolution
        bottomright *= resolution
        average_resolution = 0.5 * (resolution.x + resolution.y)
        if ratio:
            granularity = 1.0 / resolution.x / resolution.y
            collision = 0.0
            for _, point in Point.range(topleft, bottomright, resolution):
                distance1 = max(1.0 - average_resolution * self.distance(point - self.center), 0.0)
                distance2 = max(1.0 - average_resolution * other.distance(point - other.center), 0.0)
                average_distance = 0.5 * (distance1 + distance2)
                if average_distance > 0.95:
                    collision += granularity * average_distance
            collision1 = collision / self.shape.area
            collision2 = collision / other.shape.area
            if other.id is not None:
                self.collisions[other.id] = collision1
            if self.id is not None:
                other.collisions[self.id] = collision2
            if symmetric:
                return min(collision1, collision2)
            else:
                return (collision1, collision2)
        else:
            min_distance = 1.0 / average_resolution
            for _, point in Point.range(topleft, bottomright, resolution):
                if (self.distance(point - self.center) <= min_distance) and (other.distance(point - other.center) <= min_distance):
                    return True

    def not_collides(self, other, ratio=False, symmetric=False, resolution=None):
        if resolution is None:
            resolution = default_resolution
        topleft1 = self.topleft
        bottomright1 = self.bottomright
        topleft2 = other.topleft
        bottomright2 = other.bottomright
        if topleft1.x < topleft2.x and topleft1.y < topleft2.y and bottomright1.x > bottomright2.x and bottomright1.y > bottomright2.y:
            if not ratio:
                return False
            elif symmetric:
                return 0.0
            else:
                return (0.0, 0.0)
        elif bottomright1.distance(topleft1) < bottomright2.distance(topleft2):
            topleft, bottomright = topleft1, bottomright1
        else:
            topleft, bottomright = topleft2, bottomright2

        topleft *= resolution
        bottomright *= resolution
        average_resolution = 0.5 * (resolution.x + resolution.y)
        if ratio:
            granularity = 1.0 / resolution.x / resolution.y
            collision = 0.0
            for _, point in Point.range(topleft, bottomright, resolution):
                distance1 = min(average_resolution * self.distance(point - self.center), 1.0)
                distance2 = max(1.0 - average_resolution * other.distance(point - other.center), 0.0)
                average_distance = 0.5 * (distance1 + distance2)
                if average_distance > 0.95:
                    collision += granularity * average_distance
            if symmetric:
                return min(collision / self.shape.area, collision / other.shape.area)
            else:
                return (collision / self.shape.area, collision / other.shape.area)
        else:
            min_distance = 1.0 / average_resolution
            for c, point in Point.range(topleft, bottomright, resolution):
                if (self.distance(point - self.center) > min_distance) and (other.distance(point - other.center) <= min_distance):
                    return True

    @staticmethod
    def random_instance(center, rotation, size_range, distortion_range, shade_range, combination=None, combinations=None, shapes=None, colors=None, textures=None):
        # random color in texture
        rotation = random() if rotation else 0.0
        if combination is not None:
            shape, color, texture = combination
            shape = Shape.random_instance(size_range, distortion_range, shape=shape)
            color = Color.random_instance(shade_range, color=color)
            texture = Texture.random_instance([c for c in Color.colors if c != color.name], shade_range, texture=texture)
        elif combinations is not None:
            shape, color, texture = choice(combinations)
            shape = Shape.random_instance(size_range, distortion_range, shape=shape)
            color = Color.random_instance(shade_range, color=color)
            texture = Texture.random_instance([c for c in Color.colors if c != color.name], shade_range, texture=texture)
        else:
            shape = Shape.random_instance(size_range, distortion_range, shapes=shapes)
            color = Color.random_instance(shade_range, colors=colors)
            texture = Texture.random_instance([c for c in Color.colors if c != color.name], shade_range, textures=textures)
        return Entity(shape, color, texture, center, rotation)
