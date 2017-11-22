from shapeworld import util
from shapeworld.world import all_shapes, all_colors, all_textures


class WorldGenerator(object):

    MAX_ATTEMPTS = 3

    def __init__(self, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None):
        self.world_size = util.value_or_default(world_size, 64)
        self.shapes = list(util.value_or_default(shapes, all_shapes.keys()))
        self.colors = list(util.value_or_default(colors, all_colors.keys()))
        self.textures = list(util.value_or_default(textures, all_textures.keys()))
        self.world_color = util.value_or_default(world_color, 'black')
        if self.world_color in self.colors:
            self.colors.remove(self.world_color)
        self.rotation = util.value_or_default(rotation, True)
        self.size_range = util.value_or_default(size_range, (0.1, 0.25))
        self.distortion_range = util.value_or_default(distortion_range, (2.0, 3.0))  # greater than 1.0
        self.shade_range = util.value_or_default(shade_range, 0.4)
        self.collision_tolerance = util.value_or_default(collision_tolerance, 0.25)
        self.boundary_tolerance = util.value_or_default(boundary_tolerance, 0.25)

    def __str__(self):
        return self.__class__.__name__

    def initialize(self, mode):
        assert mode in (None, 'train', 'validation', 'test')
        self.mode = mode

    def __call__(self):
        if self.mode is None:
            generator = self.generate_world
        elif self.mode == 'train':
            generator = self.generate_train_world
        elif self.mode == 'validation':
            generator = self.generate_validation_world
        elif self.mode == 'test':
            generator = self.generate_test_world

        world = generator()
        return world

    def generate_world(self):
        raise NotImplementedError

    def generate_train_world(self):
        return self.generate_world()

    def generate_validation_world(self):
        return self.generate_train_world()

    def generate_test_world(self):
        return self.generate_world()


class GeneratorMixer(WorldGenerator):

    def __init__(self, generators, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        assert len(generators) >= 1
        assert all(generator.world_size == generators[0].world_size for generator in generators)
        assert not distribution or len(distribution) == len(generators)
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(distribution)
        super(GeneratorMixer, self).__init__(world_size=generators[0].world_size, world_color=generators[0].world_color, shapes=generators[0].shapes, colors=generators[0].colors, textures=generators[0].textures, rotation=generators[0].rotation, size_range=generators[0].size_range, distortion_range=generators[0].distortion_range, shade_range=generators[0].shade_range, collision_tolerance=generators[0].collision_tolerance, boundary_tolerance=generators[0].boundary_tolerance)
        self.generators = generators
        distribution = util.value_or_default(distribution, [1] * len(generators))
        self.distribution = util.cumulative_distribution(distribution)
        self.train_distribution = util.cumulative_distribution(util.value_or_default(train_distribution, distribution))
        self.validation_distribution = util.cumulative_distribution(util.value_or_default(validation_distribution, distribution))
        self.test_distribution = util.cumulative_distribution(util.value_or_default(test_distribution, distribution))

    def initialize(self, mode):
        super(GeneratorMixer, self).initialize(mode=mode)

        if mode is None:
            self.generator = util.sample(self.distribution, self.generators)
        elif mode == 'train':
            self.generator = util.sample(self.train_distribution, self.generators)
        elif mode == 'validation':
            self.generator = util.sample(self.validation_distribution, self.generators)
        elif mode == 'test':
            self.generator = util.sample(self.test_distribution, self.generators)

        self.generator.initialize(mode=mode)

    def generate_world(self):
        return self.generator.generate_world()

    def generate_train_world(self):
        return self.generator.generate_train_world()

    def generate_validation_world(self):
        return self.generator.generate_validation_world()

    def generate_test_world(self):
        return self.generator.generate_test_world()
