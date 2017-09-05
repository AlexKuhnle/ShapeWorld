from random import choice, random
from shapeworld import util
from shapeworld.world import Entity, World
from shapeworld.generators import WorldGenerator


class GenericGenerator(WorldGenerator):

    MAX_ATTEMPTS = 5

    def __init__(self, entity_counts, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, train_entity_counts=None, validation_entity_counts=None, test_entity_counts=None, validation_combinations=None, test_combinations=None, shapes_range=None, colors_range=None, textures_range=None, max_provoke_collision_rate=None):
        super(GenericGenerator, self).__init__(world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance)

        # assert world_color not random
        assert entity_counts and all(isinstance(n, int) and n >= 0 for n in entity_counts)
        assert not train_entity_counts or all(isinstance(n, int) and n >= 0 for n in train_entity_counts)
        assert not validation_entity_counts or all(isinstance(n, int) and n >= 0 for n in validation_entity_counts)
        assert not test_entity_counts or all(isinstance(n, int) and n >= 0 for n in test_entity_counts)
        self.entity_counts = entity_counts
        self.train_entity_counts = util.value_or_default(train_entity_counts, entity_counts)
        self.validation_entity_counts = util.value_or_default(validation_entity_counts, entity_counts)
        self.test_entity_counts = util.value_or_default(test_entity_counts, entity_counts)

        assert bool(validation_combinations) == bool(test_combinations)
        self.validation_combinations = util.value_or_default(validation_combinations, list())
        self.test_combinations = util.value_or_default(test_combinations, list())
        if validation_combinations is None:
            self.validation_shapes = list(self.shapes)
            self.validation_colors = list(self.colors)
            self.validation_textures = list(self.textures)
        else:
            self.validation_shapes = [shape for shape, _, _ in self.validation_combinations]
            self.validation_colors = [color for _, color, _ in self.validation_combinations]
            self.validation_textures = [texture for _, _, texture in self.validation_combinations]
        if test_combinations is None:
            self.test_shapes = list(self.shapes)
            self.test_colors = list(self.colors)
            self.test_textures = list(self.textures)
        else:
            self.test_shapes = [shape for shape, _, _ in self.test_combinations]
            self.test_colors = [color for _, color, _ in self.test_combinations]
            self.test_textures = [texture for _, _, texture in self.test_combinations]

        assert shapes_range is None or (1 <= shapes_range[0] <= shapes_range[1] <= len(self.shapes) and shapes_range[1] <= len(self.validation_shapes) and shapes_range[1] <= len(self.test_shapes))
        assert colors_range is None or (1 <= colors_range[0] <= colors_range[1] <= len(self.colors) and colors_range[1] <= len(self.validation_colors) and colors_range[1] <= len(self.test_colors))
        assert textures_range is None or (1 <= textures_range[0] <= textures_range[1] <= len(self.textures) and textures_range[1] <= len(self.validation_textures) and textures_range[1] <= len(self.test_textures))
        self.shapes_range = shapes_range
        self.colors_range = colors_range
        self.textures_range = textures_range

        assert max_provoke_collision_rate is None or (isinstance(max_provoke_collision_rate, float) and 0.0 <= max_provoke_collision_rate <= 1.0)
        self.max_provoke_collision_rate = max_provoke_collision_rate if max_provoke_collision_rate is not None else 0.5

    def sample_values(self, mode):
        super(GenericGenerator, self).sample_values(mode=mode)

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
            self.selected_shapes = util.choice(self.selected_shapes, self.shapes_range)
        if self.colors_range is not None:
            self.selected_colors = util.choice(self.selected_colors, self.colors_range)
        if self.textures_range is not None:
            self.selected_textures = util.choice(self.selected_textures, self.textures_range)

        self.provoke_collision_rate = random() * self.max_provoke_collision_rate

    def generate_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        provoke_collision = random() < self.provoke_collision_rate
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            center = world.random_location(provoke_collision=provoke_collision)
            entity = Entity.random_instance(center=center, shapes=self.selected_shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=self.selected_colors, shade_range=self.shade_range, textures=self.selected_textures)
            if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                n += 1
                provoke_collision = random() < self.provoke_collision_rate
            if n == self.num_entities:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_train_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        n = 0
        provoke_collision = random() < self.provoke_collision_rate
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            center = world.random_location(provoke_collision=provoke_collision)
            entity = Entity.random_instance(center=center, shapes=self.selected_shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=self.selected_colors, shade_range=self.shade_range, textures=self.selected_textures)
            combination = (entity.shape.name, entity.color.name, entity.texture.name)
            if combination in self.validation_combinations or combination in self.test_combinations:
                continue
            if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                n += 1
                provoke_collision = random() < self.provoke_collision_rate
            if n == self.num_entities:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_validation_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        if self.validation_combinations:
            provoke_collision = random() < self.provoke_collision_rate
            while True:
                center = world.random_location(provoke_collision=provoke_collision)
                entity = Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=self.validation_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    break
            n = 1
        else:
            n = 0
        provoke_collision = random() < self.provoke_collision_rate
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            center = world.random_location(provoke_collision=provoke_collision)
            entity = Entity.random_instance(center=center, shapes=self.selected_shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=self.selected_colors, shade_range=self.shade_range, textures=self.selected_textures)
            if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                n += 1
                provoke_collision = random() < self.provoke_collision_rate
            if n == self.num_entities:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_test_world(self):
        world = World(self.world_size, self.world_color)
        if self.num_entities == 0:
            return world
        if self.test_combinations:
            provoke_collision = random() < self.provoke_collision_rate
            while True:
                center = world.random_location(provoke_collision=provoke_collision)
                entity = Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=self.test_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    break
            n = 1
        else:
            n = 0
        provoke_collision = random() < self.provoke_collision_rate
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            center = world.random_location(provoke_collision=provoke_collision)
            entity = Entity.random_instance(center=center, shapes=self.selected_shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=self.selected_colors, shade_range=self.shade_range, textures=self.selected_textures)
            if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                n += 1
                provoke_collision = random() < self.provoke_collision_rate
            if n == self.num_entities:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world
