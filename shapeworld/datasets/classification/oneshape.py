from shapeworld.dataset import ClassificationDataset
from shapeworld.generators import GenericGenerator


class OneshapeDataset(ClassificationDataset):

    dataset_name = 'oneshape'

    def __init__(self):
        world_generator = GenericGenerator(
            entity_counts=[1]
        )
        num_classes = len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures)
        super(OneshapeDataset, self).__init__(
            world_generator=world_generator,
            num_classes=num_classes)

    def get_classes(self, world):
        return (self.world_generator.shapes.index(world.entities[0].shape.name) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(world.entities[0].color.name) * len(self.world_generator.textures) + self.world_generator.textures.index(world.entities[0].texture.name),)


dataset = OneshapeDataset
OneshapeDataset.default_config = dict()
