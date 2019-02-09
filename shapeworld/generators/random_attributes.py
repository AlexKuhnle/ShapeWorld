from random import random
from shapeworld import util
from shapeworld.world import Entity
from shapeworld.generators import GenericGenerator


class RandomAttributesGenerator(GenericGenerator):

    def __init__(
        self,
        world_size=64,
        world_colors=('black',),
        shapes=('square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'),
        colors=('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray'),
        textures=('solid',),
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=None,
        entity_counts=(1,),
        train_entity_counts=None,
        validation_entity_counts=None,
        test_entity_counts=None,
        validation_count_rate=0.5,
        test_count_rate=0.5,
        validation_combinations=None,
        test_combinations=None,
        validation_space_rate_range=(0.0, 1.0),
        test_space_rate_range=(0.0, 1.0),
        validation_combination_rate=0.5,
        test_combination_rate=0.5,
        max_provoke_collision_rate=0.33
    ):
        super(RandomAttributesGenerator, self).__init__(
            world_size=world_size,
            world_colors=world_colors,
            shapes=shapes,
            colors=colors,
            textures=textures,
            rotation=rotation,
            size_range=size_range,
            distortion_range=distortion_range,
            shade_range=shade_range,
            collision_tolerance=collision_tolerance,
            collision_shade_difference=collision_shade_difference,
            boundary_tolerance=boundary_tolerance,
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_count_rate=validation_count_rate,
            test_count_rate=test_count_rate,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            validation_space_rate_range=validation_space_rate_range,
            test_space_rate_range=test_space_rate_range,
            validation_combination_rate=validation_combination_rate,
            test_combination_rate=test_combination_rate
        )

        assert isinstance(max_provoke_collision_rate, float) and 0.0 <= max_provoke_collision_rate <= 1.0
        self.max_provoke_collision_rate = max_provoke_collision_rate

    def initialize(self, mode):
        if not super(RandomAttributesGenerator, self).initialize(mode=mode):
            return False

        self.provoke_collision_rate = random() * self.max_provoke_collision_rate

        return True

    def model(self):
        return util.merge_dicts(
            dict1=super(RandomAttributesGenerator, self).model(),
            dict2=dict(
                provoke_collision_rate=self.provoke_collision_rate
            )
        )

    def sample_entity(self, world, last_entity, combinations=None):
        if last_entity == -1:
            self.provoke_collision = random() < self.provoke_collision_rate
        elif last_entity is not None:
            self.provoke_collision = random() < self.provoke_collision_rate
        center = world.random_location(provoke_collision=self.provoke_collision)
        if combinations is None:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, shapes=self.shapes, colors=self.colors, textures=self.textures)
        else:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=combinations)
