from random import choice
from shapeworld import util
from shapeworld.world import World
from shapeworld.generators import WorldGenerator


class GenericGenerator(WorldGenerator):

    MAX_ATTEMPTS = 5

    def __init__(self, entity_counts, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, train_entity_counts=None, validation_entity_counts=None, test_entity_counts=None, validation_combinations=None, test_combinations=None):
        super(GenericGenerator, self).__init__(
            world_size=world_size,
            world_color=world_color,
            shapes=shapes,
            colors=colors,
            textures=textures,
            rotation=rotation,
            size_range=size_range,
            distortion_range=distortion_range,
            shade_range=shade_range,
            collision_tolerance=collision_tolerance,
            boundary_tolerance=boundary_tolerance
        )

        assert util.all_and_any(isinstance(n, int) and n >= 0 for n in entity_counts)
        assert train_entity_counts is None or util.all_and_any(isinstance(n, int) and n >= 0 for n in train_entity_counts)
        assert validation_entity_counts is None or util.all_and_any(isinstance(n, int) and n >= 0 for n in validation_entity_counts)
        assert test_entity_counts is None or util.all_and_any(isinstance(n, int) and n >= 0 for n in test_entity_counts)
        self.entity_counts = entity_counts
        self.train_entity_counts = util.value_or_default(train_entity_counts, entity_counts)
        self.validation_entity_counts = util.value_or_default(validation_entity_counts, entity_counts)
        self.test_entity_counts = util.value_or_default(test_entity_counts, entity_counts)

        assert (validation_combinations is None) == (test_combinations is None)
        self.validation_combinations = util.value_or_default(validation_combinations, list())
        self.test_combinations = util.value_or_default(test_combinations, list())
        if validation_combinations is None:
            self.validation_shapes = list(self.shapes)
            self.validation_colors = list(self.colors)
            self.validation_textures = list(self.textures)
        else:
            assert len(validation_combinations) > 0
            self.validation_shapes = [shape for shape, _, _ in self.validation_combinations]
            self.validation_colors = [color for _, color, _ in self.validation_combinations]
            self.validation_textures = [texture for _, _, texture in self.validation_combinations]
        if test_combinations is None:
            self.test_shapes = list(self.shapes)
            self.test_colors = list(self.colors)
            self.test_textures = list(self.textures)
        else:
            assert len(test_combinations) > 0
            self.test_shapes = [shape for shape, _, _ in self.test_combinations]
            self.test_colors = [color for _, color, _ in self.test_combinations]
            self.test_textures = [texture for _, _, texture in self.test_combinations]

    def initialize(self, mode):
        super(GenericGenerator, self).initialize(mode=mode)
        if mode is None:
            self.num_entities = choice(self.entity_counts)
        elif mode == 'train':
            self.num_entities = choice(self.train_entity_counts)
        elif mode == 'validation':
            self.num_entities = choice(self.validation_entity_counts)
        elif mode == 'test':
            self.num_entities = choice(self.test_entity_counts)

    def sample_entity(self, world, last_entity, combinations=None):
        raise NotImplementedError

    def generate_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        last_entity = -1
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            entity = self.sample_entity(world=world, last_entity=last_entity)
            if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                last_entity = entity
                n += 1
                if n == self.num_entities:
                    break
            else:
                last_entity = None
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_train_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        last_entity = -1
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            entity = self.sample_entity(world=world, last_entity=last_entity)
            combination = (entity.shape.name, entity.color.name, entity.texture.name)
            if combination in self.validation_combinations or combination in self.test_combinations:
                last_entity = None
            elif world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                last_entity = entity
                n += 1
                if n == self.num_entities:
                    break
            else:
                last_entity = None
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_validation_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        last_entity = -1
        if self.validation_combinations:
            while True:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.validation_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    last_entity = entity
                    n += 1
                    break
                else:
                    last_entity = None
        if n < self.num_entities:
            for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
                entity = self.sample_entity(world=world, last_entity=last_entity)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    last_entity = entity
                    n += 1
                    if n == self.num_entities:
                        break
                else:
                    last_entity = None
            else:
                return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_test_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        last_entity = -1
        if self.test_combinations:
            while True:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.test_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    last_entity = entity
                    n += 1
                    break
                else:
                    last_entity = None
        if n < self.num_entities:
            for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
                entity = self.sample_entity(world=world, last_entity=last_entity)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    last_entity = entity
                    n += 1
                    if n == self.num_entities:
                        break
                else:
                    last_entity = None
            else:
                return None
        if self.collision_tolerance:
            world.sort_entities()
        return world
