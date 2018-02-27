from shapeworld.dataset import ClassificationDataset
from shapeworld.generators import RandomAttributesGenerator


class Oneshape(ClassificationDataset):

    def __init__(
        self,
        world_size=64,
        boundary_tolerance=0.2,
        pixel_noise_stddev=0.0
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=[1],
            world_size=world_size,
            boundary_tolerance=boundary_tolerance
        )

        num_classes = len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures)

        super(Oneshape, self).__init__(
            world_generator=world_generator,
            num_classes=num_classes,
            pixel_noise_stddev=pixel_noise_stddev
        )

    def get_classes(self, world):
        return (self.world_generator.shapes.index(world.entities[0].shape.name) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(world.entities[0].color.name) * len(self.world_generator.textures) + self.world_generator.textures.index(world.entities[0].texture.name),)


dataset = Oneshape
