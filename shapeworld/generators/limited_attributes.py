from random import choice, random
from shapeworld.world import Entity
from shapeworld.generators import GenericGenerator


class LimitedAttributesGenerator(GenericGenerator):

    def __init__(
        self,
        world_size=64,
        world_color='black',
        shapes=None,
        colors=None,
        textures=None,
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=0.25,
        entity_counts=(1,),
        train_entity_counts=None,
        validation_entity_counts=None,
        test_entity_counts=None,
        validation_combinations=None,
        test_combinations=None,
        max_provoke_collision_rate=0.33,
        shapes_range=(2, 4),
        colors_range=(2, 4),
        textures_range=(2, 4)
    ):
        super(LimitedAttributesGenerator, self).__init__(
            world_size=world_size,
            world_color=world_color,
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
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )

        assert isinstance(max_provoke_collision_rate, float) and 0.0 <= max_provoke_collision_rate <= 1.0
        self.max_provoke_collision_rate = max_provoke_collision_rate

        assert 1 <= shapes_range[0] <= shapes_range[1] <= len(self.shapes)
        assert 1 <= colors_range[0] <= colors_range[1] <= len(self.colors)
        assert 1 <= textures_range[0] <= textures_range[1] <= len(self.textures)
        self.shapes_range = shapes_range
        self.colors_range = colors_range
        self.textures_range = textures_range

    def initialize(self, mode):
        super(LimitedAttributesGenerator, self).initialize(mode=mode)
        self.provoke_collision_rate = random() * self.max_provoke_collision_rate

        self.selected_shapes = self.shapes
        self.selected_colors = self.colors
        self.selected_textures = self.textures

        if mode is None:
            self.num_entities = choice(self.entity_counts)
        elif mode == 'train':
            self.num_entities = choice(self.train_entity_counts)
        elif mode == 'validation':
            self.num_entities = choice(self.validation_entity_counts)
            self.selected_shapes = self.validation_shapes
            self.selected_colors = self.validation_colors
            self.selected_textures = self.validation_textures
        elif mode == 'test':
            self.num_entities = choice(self.test_entity_counts)
            self.selected_shapes = self.test_shapes
            self.selected_colors = self.test_colors
            self.selected_textures = self.test_textures

        if self.shapes_range is not None:
            self.selected_shapes = util.choice(items=self.selected_shapes, num_range=self.shapes_range, auxiliary=self.shapes)
        if self.colors_range is not None:
            self.selected_colors = util.choice(items=self.selected_colors, num_range=self.colors_range, auxiliary=self.colors)
        if self.textures_range is not None:
            self.selected_textures = util.choice(items=self.selected_textures, num_range=self.textures_range, auxiliary=self.textures)

    def sample_entity(self, world, last_entity, combinations=None):
        if last_entity == -1:
            self.provoke_collision = random() < self.provoke_collision_rate
        elif last_entity is not None:
            self.provoke_collision = random() < self.provoke_collision_rate
        center = world.random_location(provoke_collision=self.provoke_collision)
        if combinations is None:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, shapes=self.selected_shapes, colors=self.selected_colors, textures=self.selected_textures)
        else:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=combinations)
