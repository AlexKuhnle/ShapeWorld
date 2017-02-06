from itertools import product
from random import choice, randint, randrange
from shapeworld.shape import Shape
from shapeworld.color import Fill, Color
from shapeworld.world import Entity, World, WorldGenerator


class GenericWorldGenerator(WorldGenerator):

    def __init__(self, world_size, shapes, size_range, distortion_range, rotation, fills, colors, shade_range, noise_range, world_background_color, entity_counts, train_entity_counts=None, validation_entity_counts=None, test_entity_counts=None, validation_shapes=None, validation_fills=None, validation_colors=None, test_shapes=None, test_fills=None, test_colors=None, train_combinations=None, shapes_range=None, fills_range=None, colors_range=None):
        super().__init__(world_size, shapes, size_range, distortion_range, rotation, fills, colors, shade_range, noise_range, world_background_color)

        assert entity_counts and all(isinstance(n, int) and n >= 0 for n in entity_counts)
        assert not train_entity_counts or all(isinstance(n, int) and n >= 0 for n in train_entity_counts)
        assert not validation_entity_counts or all(isinstance(n, int) and n >= 0 for n in validation_entity_counts)
        assert not test_entity_counts or all(isinstance(n, int) and n >= 0 for n in test_entity_counts)
        self.entity_counts = entity_counts
        self.train_entity_counts = train_entity_counts or entity_counts
        self.validation_entity_counts = validation_entity_counts or entity_counts
        self.test_entity_counts = test_entity_counts or entity_counts

        self.train_shapes = [Shape.shapes[shape] for shape in shapes]
        self.train_fills = [Fill.fills[fill] for fill in fills]
        self.train_colors = [Color.colors[color] for color in colors]

        if validation_shapes:
            assert test_shapes and not shapes_range
            self.validation_shapes = [Shape.shapes[shape] for shape in validation_shapes]
            self.test_shapes = [Shape.shapes[shape] for shape in test_shapes]
        else:
            self.validation_shapes = self.train_shapes
            self.test_shapes = self.train_shapes
            validation_shapes = test_shapes = ()

        if validation_fills:
            assert test_fills and not fills_range
            self.validation_fills = [Fill.fills[fill] for fill in validation_fills]
            self.test_fills = [Fill.fills[fill] for fill in test_fills]
        else:
            self.validation_fills = self.train_fills
            self.test_fills = self.train_fills
            validation_fills = test_fills = ()

        if validation_colors:
            assert test_colors and not colors_range
            self.validation_colors = [Color.colors[color] for color in validation_colors]
            self.test_colors = [Color.colors[color] for color in test_colors]
        else:
            self.validation_colors = self.train_colors
            self.test_colors = self.train_colors
            validation_colors = test_colors = ()

        if not train_combinations:
            train_combinations = ()
        self.validation_combinations = set(combination for combination in product(validation_shapes or shapes, validation_fills or fills, validation_colors or colors) if combination not in train_combinations)
        self.test_combinations = set(combination for combination in product(test_shapes or shapes, test_fills or fills, test_colors or colors) if combination not in train_combinations)
        self.train_combinations = set(combination for combination in product(shapes, fills, colors) if (combination not in product(validation_shapes, validation_fills, validation_colors) and combination not in product(test_shapes, test_fills, test_colors)) or combination in train_combinations)

        assert not shapes_range or (shapes_range[0] >= 1 and len(self.train_shapes) > shapes_range[1] and len(self.validation_shapes) > shapes_range[1] and len(self.test_shapes) > shapes_range[1])
        assert not fills_range or (fills_range[0] >= 1 and len(self.train_fills) > fills_range[1] and len(self.validation_fills) > fills_range[1] and len(self.test_fills) > fills_range[1])
        assert not colors_range or (colors_range[0] >= 1 and len(self.train_colors) > colors_range[1] and len(self.validation_colors) > colors_range[1] and len(self.test_colors) > colors_range[1])
        self.shapes_range = shapes_range
        self.fills_range = fills_range
        self.colors_range = colors_range

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
        world = World(world_size=self.world_size, background=self.world_background, noise_range=self.noise_range)
        n_entity = choice(self.entity_counts)
        shapes = GenericWorldGenerator.choose(self.train_shapes, self.shapes_range)
        fills = GenericWorldGenerator.choose(self.train_fills, self.fills_range)
        colors = GenericWorldGenerator.choose(self.train_colors, self.colors_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, fills=fills, colors=colors, shade_range=self.shade_range)
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None

    def generate_train_world(self):
        world = World(world_size=self.world_size, background=self.world_background, noise_range=self.noise_range)
        n_entity = choice(self.train_entity_counts)
        shapes = GenericWorldGenerator.choose(self.train_shapes, self.shapes_range)
        fills = GenericWorldGenerator.choose(self.train_fills, self.fills_range)
        colors = GenericWorldGenerator.choose(self.train_colors, self.colors_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, fills=fills, colors=colors, shade_range=self.shade_range)
            if (str(entity.shape), str(entity.fill), str(entity.fill.color)) not in self.train_combinations:
                continue
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None

    def generate_validation_world(self):
        world = World(world_size=self.world_size, background=self.world_background, noise_range=self.noise_range)
        n_entity = choice(self.validation_entity_counts)
        shapes = GenericWorldGenerator.choose(self.validation_shapes, self.shapes_range)
        fills = GenericWorldGenerator.choose(self.validation_fills, self.fills_range)
        colors = GenericWorldGenerator.choose(self.validation_colors, self.colors_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, fills=fills, colors=colors, shade_range=self.shade_range)
            if (str(entity.shape), str(entity.fill), str(entity.fill.color)) not in self.validation_combinations:
                continue
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None

    def generate_test_world(self):
        world = World(world_size=self.world_size, background=self.world_background, noise_range=self.noise_range)
        n_entity = choice(self.test_entity_counts)
        shapes = GenericWorldGenerator.choose(self.test_shapes, self.shapes_range)
        fills = GenericWorldGenerator.choose(self.test_fills, self.fills_range)
        colors = GenericWorldGenerator.choose(self.test_colors, self.colors_range)
        n = 0
        for _ in range(100 * n_entity):
            entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, fills=fills, colors=colors, shade_range=self.shade_range)
            if (str(entity.shape), str(entity.fill), str(entity.fill.color)) not in self.test_combinations:
                continue
            n += world.add_entity(entity)
            if n >= n_entity:
                return world
        return None
