from itertools import product
from random import choice, randint, randrange
from shapeworld.entity import Entity
from shapeworld.world import World
from shapeworld.generator import WorldGenerator


class GenericWorldGenerator(WorldGenerator):

    def __init__(self, world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, entity_counts, train_entity_counts=None, validation_entity_counts=None, test_entity_counts=None, validation_shapes=None, validation_colors=None, validation_textures=None, test_shapes=None, test_colors=None, test_textures=None, train_combinations=None, shapes_range=None, colors_range=None, textures_range=None):
        super().__init__(world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation)

        assert entity_counts and all(isinstance(n, int) and n >= 0 for n in entity_counts)
        assert not train_entity_counts or all(isinstance(n, int) and n >= 0 for n in train_entity_counts)
        assert not validation_entity_counts or all(isinstance(n, int) and n >= 0 for n in validation_entity_counts)
        assert not test_entity_counts or all(isinstance(n, int) and n >= 0 for n in test_entity_counts)
        self.entity_counts = entity_counts
        self.train_entity_counts = train_entity_counts or entity_counts
        self.validation_entity_counts = validation_entity_counts or entity_counts
        self.test_entity_counts = test_entity_counts or entity_counts

        if validation_shapes:
            assert test_shapes and not shapes_range
            self.validation_shapes = validation_shapes
            self.test_shapes = test_shapes
        else:
            self.validation_shapes = self.shapes
            self.test_shapes = self.shapes
            validation_shapes = test_shapes = ()

        if validation_colors:
            assert test_colors and not colors_range
            self.validation_colors = validation_colors
            self.test_colors = test_colors
        else:
            self.validation_colors = self.colors
            self.test_colors = self.colors
            validation_colors = test_colors = ()

        if validation_textures:
            assert test_textures and not textures_range
            self.validation_textures = validation_textures
            self.test_textures = test_textures
        else:
            self.validation_textures = self.textures
            self.test_textures = self.textures
            validation_textures = test_textures = ()

        if not train_combinations:
            train_combinations = ()
        self.validation_combinations = set(combination for combination in product(validation_shapes or self.shapes, validation_colors or self.colors, validation_textures or self.textures) if combination not in train_combinations)
        self.test_combinations = set(combination for combination in product(test_shapes or self.shapes, test_colors or self.colors, test_textures or self.textures) if combination not in train_combinations)
        self.train_combinations = set(combination for combination in product(self.shapes, self.colors, self.textures) if (combination not in product(validation_shapes, validation_colors, validation_textures) and combination not in product(test_shapes, test_colors, test_textures)) or combination in train_combinations)

        assert not shapes_range or (shapes_range[0] >= 1 and len(self.shapes) > shapes_range[1] and len(self.validation_shapes) > shapes_range[1] and len(self.test_shapes) > shapes_range[1])
        assert not colors_range or (colors_range[0] >= 1 and len(self.colors) > colors_range[1] and len(self.validation_colors) > colors_range[1] and len(self.test_colors) > colors_range[1])
        assert not textures_range or (textures_range[0] >= 1 and len(self.textures) > textures_range[1] and len(self.validation_textures) > textures_range[1] and len(self.test_textures) > textures_range[1])
        self.shapes_range = shapes_range
        self.colors_range = colors_range
        self.textures_range = textures_range

    @staticmethod
    def choose(available, count_range):
        if not count_range:
            return available
        count = randint(*count_range)
        chosen = []
        before = []
        n = 0
        while n < count:
            pick = randrange(len(available))
            if pick in before:
                continue
            chosen.append(available[pick])
            before.append(pick)
            n += 1
        return chosen

    def generate_world(self):
        world_color = self.world_color
        colors = self.colors
        if world_color is None:
            colors = list(colors)
            world_color = colors.pop(randrange(len(colors)))
        world = World(self.world_size, world_color, self.noise_range)
        n_entity = choice(self.entity_counts)
        shapes = GenericWorldGenerator.choose(self.shapes, self.shapes_range)
        colors = GenericWorldGenerator.choose(colors, self.colors_range)
        textures = GenericWorldGenerator.choose(self.textures, self.textures_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None

    def generate_train_world(self):
        world_color = self.world_color
        colors = self.colors
        if world_color is None:
            colors = list(colors)
            world_color = colors.pop(randrange(len(colors)))
        world = World(self.world_size, world_color, self.noise_range)
        n_entity = choice(self.train_entity_counts)
        shapes = GenericWorldGenerator.choose(self.shapes, self.shapes_range)
        colors = GenericWorldGenerator.choose(colors, self.colors_range)
        textures = GenericWorldGenerator.choose(self.textures, self.textures_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            if (str(entity.shape), str(entity.color), str(entity.texture)) not in self.train_combinations:
                continue
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None

    def generate_validation_world(self):
        world_color = self.world_color
        while world_color is None:
            world_color = choice(self.colors)
            if world_color in self.validation_colors or world_color in self.test_colors:
                world_color = None
        world = World(self.world_size, world_color, self.noise_range)
        n_entity = choice(self.validation_entity_counts)
        shapes = GenericWorldGenerator.choose(self.validation_shapes, self.shapes_range)
        colors = GenericWorldGenerator.choose(self.validation_colors, self.colors_range)
        textures = GenericWorldGenerator.choose(self.validation_textures, self.textures_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            if (str(entity.shape), str(entity.color), str(entity.texture)) not in self.validation_combinations:
                continue
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None

    def generate_test_world(self):
        world_color = self.world_color
        while world_color is None:
            world_color = choice(self.colors)
            if world_color in self.validation_colors or world_color in self.test_colors:
                world_color = None
        world = World(self.world_size, world_color, self.noise_range)
        n_entity = choice(self.test_entity_counts)
        shapes = GenericWorldGenerator.choose(self.test_shapes, self.shapes_range)
        colors = GenericWorldGenerator.choose(self.test_colors, self.colors_range)
        textures = GenericWorldGenerator.choose(self.test_textures, self.textures_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            if (str(entity.shape), str(entity.color), str(entity.texture)) not in self.test_combinations:
                continue
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None
