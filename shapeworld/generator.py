from shapeworld.point import Point
from shapeworld.color import Color


class WorldGenerator(object):

    def __init__(self, world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation):
        self.world_size = world_size
        self.world_color = Color(world_color, Color.colors[world_color], 0.0)
        if world_color is not None and world_color in colors:
            colors.remove(world_color)
        self.noise_range = noise_range
        self.shapes = shapes
        self.size_range = size_range
        self.distortion_range = distortion_range  # greater than 1.0
        self.colors = colors
        self.shade_range = shade_range
        self.textures = textures
        self.rotation = rotation

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
