from shapeworld.world import all_shapes, all_colors, all_textures


class WorldGenerator(object):

    MAX_ATTEMPTS = 10

    def __init__(self, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, noise_range=None, collision_tolerance=None, boundary_tolerance=None):
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
        self.noise_range = noise_range or 0.1
        self.boundary_tolerance = boundary_tolerance or 0.0
        self.collision_tolerance = collision_tolerance or 0.0

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
        assert False

    def generate_world(self):
        raise NotImplementedError

    def generate_train_world(self):
        return self.generate_world()

    def generate_validation_world(self):
        return self.generate_train_world()

    def generate_test_world(self):
        return self.generate_world()
