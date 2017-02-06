from collections import namedtuple
from math import cos, pi, sin
from random import random, randrange
import numpy as np
from PIL import Image
from shapeworld.point import Point
from shapeworld.shape import Shape, WorldShape
from shapeworld.color import Color, Fill, SolidFill


EntityTuple = namedtuple('EntityTuple', ('center', 'shape', 'fill', 'rotation_sin', 'rotation_cos', 'topleft', 'bottomright'))
WorldTuple = namedtuple('WorldTuple', ('world_size', 'shape', 'fill', 'noise_range', 'entities'))


class Entity(EntityTuple):
    __slots__ = ()

    def __new__(cls, center, shape, fill, rotation_angle):
        assert isinstance(center, Point)
        assert isinstance(shape, Shape)
        assert isinstance(rotation_angle, float) and 0.0 <= rotation_angle < 1.0
        assert isinstance(fill, Fill)
        rotation_sin = sin(-rotation_angle * 2.0 * pi)
        rotation_cos = cos(-rotation_angle * 2.0 * pi)
        inv_rot_sin = sin(rotation_angle * 2.0 * pi)
        inv_rot_cos = cos(rotation_angle * 2.0 * pi)
        topleft = Point.zero
        bottomright = Point.zero
        for point in shape.polygon():
            point = point.rotate(inv_rot_sin, inv_rot_cos)
            topleft = topleft.min(point)
            bottomright = bottomright.max(point)
        topleft += center
        bottomright += center
        return EntityTuple.__new__(cls, center, shape, fill, rotation_sin, rotation_cos, topleft, bottomright)

    def __contains__(self, offset):
        return self.rotate(offset) in self.shape

    def distance(self, offset):
        return self.shape.distance(self.rotate(offset))

    def rotate(self, offset):
        return offset.rotate(self.rotation_sin, self.rotation_cos)

    def draw(self, world, world_size):
        topleft = (self.topleft * world_size - Point.one).max(0)
        bottomright = (self.bottomright * world_size + Point.one).min(world_size)
        for coord, point in Point.range(topleft, bottomright, world_size):
            offset = point - self.center
            distance = self.distance(offset)
            if distance == 0.0:
                assert offset in self or type(self.shape) is WorldShape
                world[coord] = self.fill(offset)
            else:
                assert offset not in self
                distance = max(1.0 - distance * min(*world_size), 0.0)
                world[coord] = distance * self.fill(offset) + (1.0 - distance) * world[coord]

    def overlap(self, other, inside=True):
        topleft1 = self.topleft
        bottomright1 = self.bottomright
        topleft2 = other.topleft
        bottomright2 = other.bottomright
        if inside:
            if bottomright1.x < topleft2.x or topleft1.x > bottomright2.x or bottomright1.y < topleft2.y or topleft1.y > bottomright2.y:
                return None, None
            else:
                return topleft1.max(topleft2), bottomright1.min(bottomright2)
        else:
            if topleft1.x < topleft2.x and topleft1.y < topleft2.y and bottomright1.x > bottomright2.x and bottomright1.y > bottomright2.y:
                return None, None
            else:
                return topleft2, bottomright2

    def collides(self, other, world_size, inside=True, tolerance=0.0):
        topleft, bottomright = self.overlap(other, inside=inside)
        if topleft is None:
            assert bottomright is None
            return False
        assert bottomright is not None
        topleft *= world_size
        bottomright *= world_size
        granularity = 1.0 / world_size.x / world_size.y
        collision = 0
        if tolerance > 0.0:
            for point in Point.range(topleft, bottomright):
                point /= world_size
                if ((point - self.center) in self) == inside and ((point - other.center) in other):
                    collision += granularity
            return min(collision / self.shape.area(), collision / other.shape.area()) > tolerance
        else:
            for point in Point.range(topleft, bottomright):
                point = point / world_size
                if ((self.distance(point - self.center) <= granularity) == inside) and (other.distance(point - other.center) <= granularity):
                    return True

    @staticmethod
    def random_instance(center, shapes, size_range, distortion_range, rotation, fills, colors, shade_range):
        shape = Shape.random_instance(shapes, size_range, distortion_range)
        rotation_angle = random() if rotation else 0.0
        fill = Fill.random_instance(fills, colors, shade_range)
        return Entity(center, shape, fill, rotation_angle)


class World(Entity, WorldTuple):
    __slots__ = ()

    def __new__(cls, world_size, background, noise_range):
        assert isinstance(world_size, Point) and world_size > 0
        assert isinstance(background, Fill)
        assert isinstance(noise_range, float) and 0.0 <= noise_range <= 1.0
        return WorldTuple.__new__(cls, world_size, WorldShape(), background, noise_range, [])

    @property
    def center(self):
        return Point.half

    @property
    def topleft(self):
        return Point.zero

    @property
    def bottomright(self):
        return Point.one

    def rotate(self, offset):
        return offset

    def random_center(self):
        return Point.random_instance(Point.zero, Point.one)

    def add_entity(self, entity, collision_tolerance=0.0, boundary_tolerance=0.0):
        if self.collides(entity, self.world_size, inside=False, tolerance=boundary_tolerance):
            return False
        if any(entity.collides(other, self.world_size, tolerance=collision_tolerance) for other in self.entities):
            return False
        self.entities.append(entity)
        return True

    def get_world(self):
        if type(self.fill) is SolidFill:
            if np.any(a=self.fill.color.rgb):
                world = np.tile(A=self.fill(Point.zero), reps=(self.world_size.x, self.world_size.y, 1))
            else:
                world = np.zeros(shape=(self.world_size.x, self.world_size.y, 3), dtype=np.float32)
        else:
            world = np.zeros(shape=(self.world_size.x, self.world_size.y, 3), dtype=np.float32)
            self.draw(world=world, world_size=self.world_size - 1)
        for entity in self.entities:
            entity.draw(world=world, world_size=self.world_size - 1)
        if self.noise_range > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.noise_range, size=(self.world_size.x, self.world_size.y, 3))
            mask = (noise < -self.noise_range) + (noise > self.noise_range)
            while np.any(a=mask):
                noise -= mask * noise
                noise += mask * np.random.normal(loc=0.0, scale=self.noise_range, size=(self.world_size.x, self.world_size.y, 3))
                mask = (noise < -self.noise_range) + (noise > self.noise_range)
            world += noise
            np.clip(world, a_min=0.0, a_max=1.0, out=world)
        return world

    @staticmethod
    def get_image(world):
        world = (world * 255.0).astype(dtype=np.uint8).transpose(1, 0, 2)
        return Image.fromarray(obj=world, mode='RGB')


class WorldGenerator(object):

    def __init__(self, world_size, shapes, size_range, distortion_range, rotation, fills, colors, shade_range, noise_range, world_background_color):
        self.world_size = Point(*world_size)
        if world_background_color == 'random':
            world_background_color = randrange(len(colors))
        else:
            world_background_color = colors.index(world_background_color)
        world_background_color = Color.colors[colors.pop(world_background_color)]
        self.world_background = SolidFill(Color(world_background_color[0], world_background_color[1], 0.0))
        self.shapes = {shape: Shape.shapes[shape] for shape in shapes}
        self.size_range = size_range
        self.distortion_range = distortion_range
        self.rotation = rotation
        self.fills = {fill: Fill.fills[fill] for fill in fills}
        self.colors = {color: Color.colors[color] for color in colors}
        self.shade_range = shade_range
        self.noise_range = noise_range

    def __call__(self, mode=None):
        assert mode in (None, 'train', 'validation', 'test')
        if mode is None:
            generator = self.generate_world
        elif mode == 'train':
            generator = self.generate_train_world
        elif mode == 'validation':
            generator = self.generate_validation_world
        elif mode == 'test':
            generator = self.generate_test_world
        for _ in range(10):
            world = generator()
            if world is not None:
                return world
        assert False

    def generate_world(self):
        raise NotImplementedError

    def generate_train_world(self):
        return self.generate_world()

    def generate_validation_world(self):
        return self.generate_train_world()

    def generate_test_world(self):
        return self.generate_world()
