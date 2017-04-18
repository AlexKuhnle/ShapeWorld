from random import choice, randint, randrange
from shapeworld.world import Entity, World
from shapeworld import WorldGenerator


class GenericGenerator(WorldGenerator):

    MAX_ATTEMPTS = 10

    def __init__(self, entity_counts, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, noise_range=None, collision_tolerance=None, boundary_tolerance=None, train_entity_counts=None, validation_entity_counts=None, test_entity_counts=None, validation_combinations=None, test_combinations=None, shapes_range=None, colors_range=None, textures_range=None):
        super(GenericGenerator, self).__init__(world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, noise_range, collision_tolerance, boundary_tolerance)

        # assert world_color not random
        assert entity_counts and all(isinstance(n, int) and n >= 0 for n in entity_counts)
        assert not train_entity_counts or all(isinstance(n, int) and n >= 0 for n in train_entity_counts)
        assert not validation_entity_counts or all(isinstance(n, int) and n >= 0 for n in validation_entity_counts)
        assert not test_entity_counts or all(isinstance(n, int) and n >= 0 for n in test_entity_counts)
        self.entity_counts = entity_counts
        self.train_entity_counts = train_entity_counts or entity_counts
        self.validation_entity_counts = validation_entity_counts or entity_counts
        self.test_entity_counts = test_entity_counts or entity_counts

        assert bool(validation_combinations) == bool(test_combinations)
        self.validation_combinations = list(validation_combinations or ())
        self.validation_shapes = list(shape for shape, _, _ in self.validation_combinations) or self.shapes
        self.validation_colors = list(color for _, color, _ in self.validation_combinations) or self.colors
        self.validation_textures = list(texture for _, _, texture in self.validation_combinations) or self.textures
        self.test_combinations = list(test_combinations or ())
        self.test_shapes = list(shape for shape, _, _ in self.test_combinations) or self.shapes
        self.test_colors = list(color for _, color, _ in self.test_combinations) or self.colors
        self.test_textures = list(texture for _, _, texture in self.test_combinations) or self.textures

        assert not shapes_range or (1 <= shapes_range[0] <= shapes_range[1] <= len(self.shapes) and shapes_range[1] <= len(self.validation_shapes) and shapes_range[1] <= len(self.test_shapes))
        assert not colors_range or (1 <= colors_range[0] <= colors_range[1] <= len(self.colors) and colors_range[1] <= len(self.validation_colors) and colors_range[1] <= len(self.test_colors))
        assert not textures_range or (1 <= textures_range[0] <= textures_range[1] <= len(self.textures) and textures_range[1] <= len(self.validation_textures) and textures_range[1] <= len(self.test_textures))
        self.shapes_range = shapes_range
        self.colors_range = colors_range
        self.textures_range = textures_range

    @staticmethod
    def choose(available, count_range):
        if not count_range:
            return available
        count = randint(*count_range)
        chosen = []
        before = []
        n = 0
        while n < count:
            pick = randrange(len(available))
            if pick in before:
                continue
            chosen.append(available[pick])
            before.append(pick)
            n += 1
        return chosen

    def generate_world(self):
        world = World(self.world_size, self.world_color, self.noise_range)
        n_entity = choice(self.entity_counts)
        if n_entity == 0:
            return world
        shapes = GenericGenerator.choose(self.shapes, self.shapes_range)
        colors = GenericGenerator.choose(self.colors, self.colors_range)
        textures = GenericGenerator.choose(self.textures, self.textures_range)
        n = 0
        for _ in range(n_entity * GenericGenerator.MAX_ATTEMPTS):
            entity = Entity.random_instance(center=world.random_location(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            n += world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance)
            if n >= n_entity:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_train_world(self):
        world = World(self.world_size, self.world_color, self.noise_range)
        n_entity = choice(self.train_entity_counts)
        if n_entity == 0:
            return world
        shapes = GenericGenerator.choose(self.shapes, self.shapes_range)
        colors = GenericGenerator.choose(self.colors, self.colors_range)
        textures = GenericGenerator.choose(self.textures, self.textures_range)
        n = 0
        for _ in range(n_entity * GenericGenerator.MAX_ATTEMPTS):
            entity = Entity.random_instance(center=world.random_location(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            combination = (str(entity.shape), str(entity.color), str(entity.texture))
            if combination in self.validation_combinations or combination in self.test_combinations:
                continue
            n += world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance)
            if n >= n_entity:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_validation_world(self):
        world = World(self.world_size, self.world_color, self.noise_range)
        n_entity = choice(self.validation_entity_counts)
        if n_entity == 0:
            return world
        shapes = GenericGenerator.choose(self.validation_shapes, self.shapes_range)
        colors = GenericGenerator.choose(self.validation_colors, self.colors_range)
        textures = GenericGenerator.choose(self.validation_textures, self.textures_range)
        if self.validation_combinations:
            for _ in range(GenericGenerator.MAX_ATTEMPTS):
                entity = Entity.random_instance(center=world.random_location(), rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=self.validation_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    break
            else:
                return None
            n = 1
        else:
            n = 0
        for _ in range(n_entity * GenericGenerator.MAX_ATTEMPTS):
            entity = Entity.random_instance(center=world.random_location(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            n += world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance)
            if n >= n_entity:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world

    def generate_test_world(self):
        world = World(self.world_size, self.world_color, self.noise_range)
        n_entity = choice(self.test_entity_counts)
        if n_entity == 0:
            return world
        shapes = GenericGenerator.choose(self.test_shapes, self.shapes_range)
        colors = GenericGenerator.choose(self.test_colors, self.colors_range)
        textures = GenericGenerator.choose(self.test_textures, self.textures_range)
        if self.test_combinations:
            for _ in range(GenericGenerator.MAX_ATTEMPTS):
                entity = Entity.random_instance(center=world.random_location(), rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=self.test_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance):
                    break
            else:
                return None
            n = 1
        else:
            n = 0
        for _ in range(n_entity * GenericGenerator.MAX_ATTEMPTS):
            entity = Entity.random_instance(center=world.random_location(), shapes=shapes, size_range=self.size_range, distortion_range=self.distortion_range, rotation=self.rotation, colors=colors, shade_range=self.shade_range, textures=textures)
            n += world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance)
            if n >= n_entity:
                break
        else:
            return None
        if self.collision_tolerance:
            world.sort_entities()
        return world
