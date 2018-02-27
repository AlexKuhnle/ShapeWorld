from shapeworld.dataset import ClassificationDataset
from shapeworld.generators import ReinforcedAttributesGenerator


class Countshape(ClassificationDataset):

    def __init__(
        self,
        world_size=64,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        max_provoke_collision_rate=0.33,
        collision_tolerance=0.2,
        boundary_tolerance=0.2,
        pixel_noise_stddev=0.0
    ):

        world_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 3),
            entity_counts=entity_counts,
            world_size=world_size,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            max_provoke_collision_rate=max_provoke_collision_rate,
            collision_tolerance=collision_tolerance,
            boundary_tolerance=boundary_tolerance
        )

        num_classes = len(world_generator.shapes) * len(world_generator.colors) * len(world_generator.textures)

        super(Countshape, self).__init__(
            world_generator=world_generator,
            num_classes=num_classes,
            multi_class=True,
            class_count=True,
            pixel_noise_stddev=pixel_noise_stddev
        )

    def get_classes(self, world):
        return [self.world_generator.shapes.index(entity.shape.name) * len(self.world_generator.colors) * len(self.world_generator.textures) + self.world_generator.colors.index(entity.color.name) * len(self.world_generator.textures) + self.world_generator.textures.index(entity.texture.name) for entity in world.entities]


dataset = Countshape
