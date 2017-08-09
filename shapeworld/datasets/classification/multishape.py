from shapeworld.dataset import ClassificationDataset
from shapeworld.generators import GenericGenerator


class MultiShapeDataset(ClassificationDataset):

    dataset_name = 'multishape'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, class_count, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, **kwargs):
        world_generator = GenericGenerator(entity_counts, world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts)
        num_classes = len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures)
        super(MultiShapeDataset, self).__init__(
            world_generator=world_generator,
            num_classes=num_classes,
            multi_class=True,
            class_count=class_count)

    def get_classes(self, world):
        return [self.world_generator.shapes.index(world.entities[0].shape.name) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(world.entities[0].color.name) * len(self.world_generator.textures) + self.world_generator.textures.index(world.entities[0].texture.name) for entity in world.entities]


dataset = MultiShapeDataset
MultiShapeDataset.default_config = {
    'entity_counts': [1, 2, 3, 4, 5, 6],
    'train_entity_counts': [1, 2, 3, 5],
    'validation_entity_counts': [4],
    'test_entity_counts': [6],
    'class_count': False
}
