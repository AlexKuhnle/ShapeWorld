from shapeworld.dataset import ClassificationDataset
from shapeworld.generators import GenericGenerator


class OneShapeDataset(ClassificationDataset):

    dataset_name = 'oneshape'

    def __init__(self, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, **kwargs):
        world_generator = GenericGenerator([1], world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance)
        num_classes = len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures)
        super(OneShapeDataset, self).__init__(
            world_generator=world_generator,
            num_classes=num_classes)

    def get_classes(self, world):
        return (self.world_generator.shapes.index(str(world.entities[0].shape)) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(str(world.entities[0].color)) * len(self.world_generator.textures) + self.world_generator.textures.index(str(world.entities[0].texture)),)


dataset = OneShapeDataset
OneShapeDataset.default_config = dict()
