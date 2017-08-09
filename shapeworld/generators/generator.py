from shapeworld.util import cumulative_distribution, sample
from shapeworld.world import all_shapes, all_colors, all_textures


class WorldGenerator(object):

    MAX_ATTEMPTS = 10
    name = None

    def __init__(self, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None):
        assert self.__class__.name
        self.world_size = world_size or 64
        self.shapes = shapes or list(all_shapes.keys())
        self.colors = colors or list(all_colors.keys())
        self.textures = textures or list(all_textures.keys())
        self.world_color = world_color or 'black'
        if self.world_color in self.colors:
            self.colors.remove(self.world_color)
        self.rotation = rotation if rotation is not None else True
        self.size_range = size_range or (0.15, 0.3)
        self.distortion_range = distortion_range or (2.0, 3.0)  # greater than 1.0
        self.shade_range = shade_range or 0.5
        self.collision_tolerance = collision_tolerance or 0.0
        self.boundary_tolerance = boundary_tolerance or 0.0

    def __str__(self):
        return self.__class__.name

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
        for _ in range(WorldGenerator.MAX_ATTEMPTS):
            world = generator()
            if world is not None:
                return world
        return None

    def generate_world(self):
        raise NotImplementedError

    def generate_train_world(self):
        return self.generate_world()

    def generate_validation_world(self):
        return self.generate_train_world()

    def generate_test_world(self):
        return self.generate_world()


class GeneratorMixer(object):

    name = 'mixer'

    def __init__(self, generators, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        assert len(generators) >= 1
        assert all(generator.world_size == generators[0].world_size for generator in generators)
        self.generators = generators
        super(GeneratorMixer, self).__init__(world_size=generators[0].world_size, world_color=generators[0].world_color, shapes=generators[0].shapes, colors=generators[0].colors, textures=generators[0].textures, rotation=generators[0].rotation, size_range=generators[0].size_range, distortion_range=generators[0].distortion_range, shade_range=generators[0].shade_range, collision_tolerance=generators[0].collision_tolerance, boundary_tolerance=generators[0].boundary_tolerance)
        assert not distribution or len(distribution) == len(generators)
        self.distribution = cumulative_distribution(distribution or [1] * len(generators))
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(self.distribution)
        self.train_distribution = cumulative_distribution(train_distribution) if train_distribution else self.distribution
        self.validation_distribution = cumulative_distribution(validation_distribution) if validation_distribution else self.distribution
        self.test_distribution = cumulative_distribution(test_distribution) if test_distribution else self.distribution

    def __call__(self, mode=None):
        assert mode in (None, 'train', 'validation', 'test')
        if mode is None:
            generator = sample(self.distribution, self.generators)
            generator = generator.generate_world
        elif mode == 'train':
            generator = sample(self.train_distribution, self.generators)
            generator = generator.generate_train_world
        elif mode == 'validation':
            generator = sample(self.validation_distribution, self.generators)
            generator = generator.generate_validation_world
        elif mode == 'test':
            generator = sample(self.test_distribution, self.generators)
            generator = generator.generate_test_world
        for _ in range(WorldGenerator.MAX_ATTEMPTS):
            world = generator()
            if world is not None:
                return world
        return None

    def generate_world(self):
        generator = sample(self.distribution, self.captioners)
        return generator.generate_world()

    def generate_train_world(self):
        generator = sample(self.train_distribution, self.captioners)
        return generator.generate_world()

    def generate_validation_world(self):
        generator = sample(self.validation_distribution, self.captioners)
        return generator.generate_world()

    def generate_test_world(self):
        generator = sample(self.test_distribution, self.captioners)
        return generator.generate_world()
