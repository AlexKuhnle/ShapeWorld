from __future__ import division
from random import choice, random
import numpy as np
from PIL import Image
from shapeworld import util
from shapeworld.util import Point
from shapeworld.world import Entity, Color
from shapeworld.world.shape import WorldShape
from shapeworld.world.texture import SolidTexture


class World(Entity):

    COLLISION_MIN_DISTANCE = 0.75

    __slots__ = ('size', 'entities', 'shape', 'color', 'texture', 'center', 'rotation', 'rotation_sin', 'rotation_cos', 'relative_topleft', 'relative_bottomright', 'topleft', 'bottomright')

    def __init__(self, size, color):
        assert isinstance(size, int) and size > 0
        assert isinstance(color, str) and color in Color.colors
        super(World, self).__init__(WorldShape(), Color(color, Color.colors[color], 0.0), SolidTexture(), Point.half, 0.0)
        self.relative_topleft = -Point.half
        self.relative_bottomright = Point.half
        self.topleft = Point.zero
        self.bottomright = Point.one
        self.size = Point(size, size)
        self.entities = []

    def __eq__(self, other):
        raise NotImplementedError

    def model(self):
        return {'size': self.size.x, 'color': self.color.model(), 'entities': [entity.model() for entity in self.entities]}

    @staticmethod
    def from_model(model):
        world = World(size=model['size'], color=Color.from_model(model['color']))
        for entity_model in model['entities']:
            world.entities.append(Entity.from_model(entity_model))
        return world

    def copy(self, include_entities=True):
        copy = World(size=self.size.x, color=str(self.color))
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

    def draw(self, world_array, world_size):
        for entity in self.entities:
            entity.draw(world_array=world_array, world_size=world_size)

    def random_location(self, provoke_collision=False):
        if provoke_collision and self.entities:
            entity = choice(self.entities)
            angle = Point.from_angle(angle=random())
            return entity.center + angle * entity.shape.size * (self.__class__.COLLISION_MIN_DISTANCE + random())
        else:
            return Point.random_instance(Point.zero, Point.one)

    def add_entity(self, entity, boundary_tolerance=0.0, collision_tolerance=0.0):
        entity.id = len(self.entities)
        if boundary_tolerance > 0.0:
            if self.not_collides(entity, ratio=True, resolution=self.size)[1] > boundary_tolerance:
                return False
        else:
            if self.not_collides(entity, resolution=self.size):
                return False
        if collision_tolerance > 0.0:
            for other in self.entities:
                collision = entity.collides(other, ratio=True, symmetric=True, resolution=self.size)
                if collision > collision_tolerance or (collision > 0.0 and entity.color == other.color):
                    # can't distinguish shapes of same color
                    return False
                if entity.overall_collision() > collision_tolerance:
                    return False
        else:
            if any(entity.collides(other, resolution=self.size) for other in self.entities):
                return False
        self.entities.append(entity)
        return True

    def sort_entities(self):
        contained = {n: set() for n in range(len(self.entities))}
        for n in range(len(self.entities)):
            entity1 = self.entities[n]
            for k in range(n + 1, len(self.entities)):
                entity2 = self.entities[k]
                c1, c2 = entity1.collides(entity2, ratio=True, symmetric=False, resolution=self.size)
                if c2 > c1:
                    contained[n].add(k)
                elif c1 > c2 or c1 != 0.0:
                    contained[k].add(n)
        sort_indices = util.toposort(partial_order=contained)
        self.entities = [self.entities[n] for n in sort_indices]
        for n, entity in enumerate(self.entities):
            entity.id = n
            entity.collisions = {sort_indices.index(i): c for i, c in entity.collisions.items()}

    def get_array(self, noise_range=None):
        color = self.color.get_color()
        if not color.any():
            world_array = np.zeros(shape=(self.size.y, self.size.x, 3), dtype=np.float32)
        else:
            world_array = np.tile(A=np.array(object=color, dtype=np.float32), reps=(self.size.x, self.size.y, 1))
        self.draw(world_array=world_array, world_size=self.size)
        if noise_range is not None and noise_range > 0.0:
            noise = np.random.normal(loc=0.0, scale=noise_range, size=(self.size.y, self.size.x, 3))
            mask = (noise < -2.0 * noise_range) + (noise > 2.0 * noise_range)
            while np.any(a=mask):
                noise -= mask * noise
                noise += mask * np.random.normal(loc=0.0, scale=noise_range, size=(self.size.y, self.size.x, 3))
                mask = (noise < -2.0 * noise_range) + (noise > 2.0 * noise_range)
            world_array += noise
            np.clip(world_array, a_min=0.0, a_max=1.0, out=world_array)
        return world_array

    @staticmethod
    def get_image(world_array):
        image = Image.fromarray(obj=(world_array * 255.0).astype(dtype=np.uint8), mode='RGB')
        return image

    @staticmethod
    def from_image(image):
        world_array = (np.array(object=image, dtype=np.float32) / 255.0)
        if world_array.shape[2] == 4:
            world_array = world_array[:, :, :3]
        assert world_array.shape[2] == 3
        return world_array
