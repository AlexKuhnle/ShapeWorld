from shapeworld.dataset import ClassificationDataset
from shapeworld.generators.generic import GenericWorldGenerator


class CountShapeDataset(ClassificationDataset):

    multi_class_flag = True
    class_count_flag = True

    def __init__(self, world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, **kwargs):
        world_generator = GenericWorldGenerator(world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, entity_counts, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts)
        super().__init__(
            world_generator=world_generator,
            num_classes=len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures))
        self.shape_names = sorted(self.world_generator.shapes.keys())
        self.color_names = sorted(self.world_generator.colors.keys())

    def get_classes(self, world):
        return [self.world_generator.shapes.index(entity.shape) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(entity.color) * len(self.world_generator.textures) + self.world_generator.textures.index(entity.texture) for entity in world.entities]


dataset = CountShapeDataset
CountShapeDataset.name = 'CountShape'
CountShapeDataset.default_config = {
    'world_size': 64,
    'world_color': 'black',
    'noise_range': 0.1,
    'shapes': ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'],
    'size_range': [0.2, 0.3],
    'distortion_range': [2.0, 3.0],
    'colors': ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white'],
    'shade_range': 0.5,
    'textures': ['solid'],
    'rotation': True,
    'entity_counts': [1, 2, 3, 4, 5, 6],
    'train_entity_counts': [1, 2, 3, 4],
    'validation_entity_counts': [5],
    'test_entity_counts': [6]}
