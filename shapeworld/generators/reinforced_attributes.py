from random import randint, random
from shapeworld import util
from shapeworld.world import Entity
from shapeworld.generators import GenericGenerator


class ReinforcedAttributesGenerator(GenericGenerator):

    def __init__(self, reinforcement_range, entity_counts, train_entity_counts=None, validation_entity_counts=None, test_entity_counts=None, validation_combinations=None, test_combinations=None, max_provoke_collision_rate=None, **kwargs):
        super(ReinforcedAttributesGenerator, self).__init__(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            **kwargs
        )

        assert 0 <= reinforcement_range[0] <= reinforcement_range[1]
        self.reinforcement_range = reinforcement_range

        assert max_provoke_collision_rate is None or isinstance(max_provoke_collision_rate, float) and 0.0 <= max_provoke_collision_rate <= 1.0
        self.max_provoke_collision_rate = util.value_or_default(max_provoke_collision_rate, 0.5)

    def initialize(self, mode):
        super(ReinforcedAttributesGenerator, self).initialize(mode=mode)
        self.provoke_collision_rate = random() * self.max_provoke_collision_rate
        self.shape_pool = list(self.shapes)
        self.color_pool = list(self.colors)
        self.texture_pool = list(self.textures)

    def sample_entity(self, world, last_entity, combinations=None):
        if last_entity == -1:
            self.provoke_collision = random() < self.provoke_collision_rate
        elif last_entity is not None:
            self.provoke_collision = random() < self.provoke_collision_rate
            reinforcement_step = randint(*self.reinforcement_range)
            for _ in range(reinforcement_step):
                self.shape_pool.append(last_entity.shape.name)
                self.color_pool.append(last_entity.color.name)
                self.texture_pool.append(last_entity.texture.name)
        center = world.random_location(provoke_collision=self.provoke_collision)
        if combinations is None:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, shapes=self.shape_pool, colors=self.color_pool, textures=self.texture_pool)
        else:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=combinations)
