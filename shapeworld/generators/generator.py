from random import choice
from shapeworld import util


class WorldGenerator(object):

    MAX_SAMPLE_ATTEMPTS = 10
    MAX_ATTEMPTS = 5

    def __init__(
        self,
        world_size=64,
        world_colors=('black',),
        shapes=('square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'),
        colors=('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray'),
        textures=('solid',),
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=None
    ):
        self.world_size = world_size
        self.world_colors = list(world_colors)
        self.shapes = list(shapes)
        self.all_colors = list(colors)
        self.textures = list(textures)
        self.rotation = rotation
        self.size_range = size_range
        self.distortion_range = distortion_range
        self.shade_range = shade_range
        self.collision_tolerance = collision_tolerance
        self.collision_shade_difference = collision_shade_difference
        self.boundary_tolerance = util.value_or_default(boundary_tolerance, self.collision_tolerance)

    def __str__(self):
        return self.__class__.__name__

    def initialize(self, mode):
        assert mode in (None, 'train', 'validation', 'test')
        self.mode = mode
        self.world_color = choice(self.world_colors)
        self.colors = [color for color in self.all_colors if color != self.world_color]
        return True

    def model(self):
        return dict(
            name=str(self),
            mode=self.mode,
            world_color=self.world_color
        )

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

    def __init__(
        self,
        generators,
        distribution=None,
        train_distribution=None,
        validation_distribution=None,
        test_distribution=None
    ):
        assert len(generators) >= 1
        assert all(generator.world_size == generators[0].world_size for generator in generators)
        assert not distribution or len(distribution) == len(generators)
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(distribution)
        super(GeneratorMixer, self).__init__(world_size=generators[0].world_size, world_colors=generators[0].world_colors, shapes=generators[0].shapes, colors=generators[0].all_colors, textures=generators[0].textures, rotation=generators[0].rotation, size_range=generators[0].size_range, distortion_range=generators[0].distortion_range, shade_range=generators[0].shade_range, collision_tolerance=generators[0].collision_tolerance, boundary_tolerance=generators[0].boundary_tolerance)
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

        return self.generator.initialize(mode=mode)

    def generate_world(self):
        return self.generator.generate_world()

    def generate_train_world(self):
        return self.generator.generate_train_world()

    def generate_validation_world(self):
        return self.generator.generate_validation_world()

    def generate_test_world(self):
        return self.generator.generate_test_world()
