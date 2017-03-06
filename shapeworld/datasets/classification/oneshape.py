from shapeworld.dataset import ClassificationDataset
from shapeworld.generators.generic import GenericWorldGenerator


class OneShapeDataset(ClassificationDataset):

    def __init__(self, world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, **kwargs):
        world_generator = GenericWorldGenerator(world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, [1])
        super().__init__(
            world_generator=world_generator,
            num_classes=len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures))

    def get_classes(self, world):
        return (self.world_generator.shapes.index(world.entities[0].shape) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(world.entities[0].color) * len(self.world_generator.textures) + self.world_generator.textures.index(world.entities[0].texture),)


dataset = OneShapeDataset
OneShapeDataset.name = 'OneShape'
OneShapeDataset.default_config = {
    'world_size': 64,
    'world_color': 'black',
    'noise_range': 0.1,
    'shapes': ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'],
    'size_range': [0.2, 0.3],
    'distortion_range': [2.0, 3.0],
    'colors': ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white'],
    'shade_range': 0.5,
    'textures': ['solid'],
    'rotation': True}
