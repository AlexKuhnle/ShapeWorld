import numpy as np
from PIL import Image
from shapeworld.point import Point
from shapeworld.shape import WorldShape
from shapeworld.color import Color
from shapeworld.texture import SolidTexture
from shapeworld.entity import Entity


class World(Entity):
    __slots__ = ('size', 'noise_range', 'entities', 'shape', 'color', 'texture', 'center', 'rotation', 'rotation_sin', 'rotation_cos', 'topleft', 'bottomright')

    def __init__(self, size, color, noise_range):
        assert isinstance(size, int) and size > 0
        assert isinstance(color, Color)
        assert isinstance(noise_range, float) and 0.0 <= noise_range <= 1.0
        super().__init__(WorldShape(), color, SolidTexture(), Point.half, 0.0)
        self.topleft = Point.zero
        self.bottomright = Point.one
        self.size = Point(size, size)
        self.noise_range = noise_range
        self.entities = []

    def __str__(self):
        return 'world'

    def model(self):
        return {'size': self.size.x, 'color': self.color.model(), 'noise_range': self.noise_range, 'entities': [entity.model() for entity in self.entities]}

    def rotate(self, offset):
        return offset

    def __contains__(self, offset):
        return offset in self.shape

    def distance(self, offset):
        return self.shape.distance(offset)

    def draw(self, world, world_size):
        for entity in self.entities:
            entity.draw(world=world, world_size=world_size)

    def random_center(self):
        return Point.random_instance(Point.zero, Point.one)

    def add_entity(self, entity, collision_tolerance=0.0, boundary_tolerance=0.0):
        if self.not_collides(entity, self.size, tolerance=boundary_tolerance):
            return False
        if any(entity.collides(other, self.size, tolerance=collision_tolerance) for other in self.entities):
            return False
        self.entities.append(entity)
        return True

    def get_world(self, noise=True):
        if self.color == 'black':
            world = np.zeros(shape=(self.size.x, self.size.y, 3), dtype=np.float32)
        else:
            world = np.tile(A=np.asarray(a=Color.colors[self.color], dtype=np.float32), reps=self.size)
        self.draw(world=world, world_size=self.size)
        if noise and self.noise_range > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.noise_range, size=(self.size.x, self.size.y, 3))
            mask = (noise < -self.noise_range) + (noise > self.noise_range)
            while np.any(a=mask):
                noise -= mask * noise
                noise += mask * np.random.normal(loc=0.0, scale=self.noise_range, size=(self.size.x, self.size.y, 3))
                mask = (noise < -self.noise_range) + (noise > self.noise_range)
            world += noise
            np.clip(world, a_min=0.0, a_max=1.0, out=world)
        return world

    @staticmethod
    def get_image(world):  # world matrix
        image = Image.fromarray(obj=(world * 255.0).astype(dtype=np.uint8).transpose(1, 0, 2), mode='RGB')
        return image

    @staticmethod
    def from_image(image):  # world matrix
        world = (np.asarray(a=image, dtype=np.float32) / 255.0).transpose(1, 0, 2)
        return world
