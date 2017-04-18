import numpy as np
from PIL import Image
from shapeworld.util import toposort, Point
from shapeworld.world import Entity
from shapeworld.world.shape import WorldShape
from shapeworld.world.color import Color
from shapeworld.world.texture import SolidTexture


class World(Entity):

    __slots__ = ('size', 'noise_range', 'entities', 'shape', 'color', 'texture', 'center', 'rotation', 'rotation_sin', 'rotation_cos', 'topleft', 'bottomright')

    def __init__(self, size, color, noise_range):
        assert isinstance(size, int) and size > 0
        assert isinstance(color, str) and color in Color.colors
        assert isinstance(noise_range, float) and 0.0 <= noise_range <= 1.0
        super(World, self).__init__(WorldShape(), Color(color, Color.colors[color], 0.0), SolidTexture(), Point.half, 0.0)
        self.topleft = Point.zero
        self.bottomright = Point.one
        self.size = Point(size, size)
        self.noise_range = noise_range
        self.entities = []

    def model(self):
        return {'size': self.size.x, 'color': self.color.model(), 'noise-range': self.noise_range, 'entities': [entity.model() for entity in self.entities]}

    def copy(self, include_entities=True):
        copy = World(size=self.size.x, color=str(self.color), noise_range=self.noise_range)
        if include_entities:
            for entity in self.entities:
                copy.entities.append(entity.copy())
        return copy

    def rotate(self, offset):
        return offset

    def __contains__(self, offset):
        return offset in self.shape

    def distance(self, offset):
        return self.shape.distance(offset)

    def draw(self, world, world_size):
        for entity in self.entities:
            entity.draw(world=world, world_size=world_size)

    def random_location(self):
        return Point.random_instance(Point.zero, Point.one)

    def add_entity(self, entity, boundary_tolerance=0.0, collision_tolerance=0.0):
        if boundary_tolerance:
            if self.not_collides(entity, self.size, ratio=True) > boundary_tolerance:
                return False
        else:
            if self.not_collides(entity, self.size):
                return False
        if collision_tolerance:
            if any(entity.collides(other, self.size, ratio=True) > collision_tolerance for other in self.entities):
                return False
        else:
            if any(entity.collides(other, self.size) for other in self.entities):
                return False
        entity.id = len(self.entities)
        self.entities.append(entity)
        return True

    def sort_entities(self):
        contained = {n: set() for n in range(len(self.entities))}
        for n in range(len(self.entities)):
            entity1 = self.entities[n]
            for k in range(n + 1, len(self.entities)):
                entity2 = self.entities[k]
                c1, c2 = entity1.collides(entity2, self.size, ratio=True, symmetric=False)
                if c2 > c1:
                    contained[n].add(k)
                elif c1 > c2:
                    contained[k].add(n)
        self.entities = [self.entities[n] for n in toposort(partial_order=contained)]
        for n, entity in enumerate(self.entities):
            entity.id = n

    def get_world(self, noise=True):
        color = self.color.get_color()
        if not color.any():
            world = np.zeros(shape=(self.size.x, self.size.y, 3), dtype=np.float32)
        else:
            world = np.tile(A=np.array(object=color, dtype=np.float32), reps=(self.size.x, self.size.y, 1))
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
        world = (np.array(object=image, dtype=np.float32) / 255.0).transpose(1, 0, 2)
        return world
