from shapeworld.dataset import ClassificationDataset
from shapeworld.generators import RandomAttributesGenerator


class Multishape(ClassificationDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,)
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts
        )

        num_classes = len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures)

        super(Multishape, self).__init__(
            world_generator=world_generator,
            num_classes=num_classes,
            multi_class=True,
            class_count=False)

    def get_classes(self, world):
        return {self.world_generator.shapes.index(entity.shape.name) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(entity.color.name) * len(self.world_generator.textures) + self.world_generator.textures.index(entity.texture.name) for entity in world.entities}


dataset = Multishape
