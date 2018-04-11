from random import choice, random, uniform
from shapeworld import util
from shapeworld.world import World
from shapeworld.generators import WorldGenerator


class GenericGenerator(WorldGenerator):

    MAX_ATTEMPTS = 5

    def __init__(
        self,
        world_size=64,
        world_color='black',
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
        validation_count_rate=0.5,
        test_entity_counts=None,
        test_count_rate=0.5,
        validation_combinations=None,
        validation_space_rate_range=(0.0, 1.0),
        validation_combination_rate=0.5,
        test_combinations=None,
        test_space_rate_range=(0.0, 1.0),
        test_combination_rate=0.5
    ):
        super(GenericGenerator, self).__init__(
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
            boundary_tolerance=boundary_tolerance
        )

        assert util.all_and_any(isinstance(n, int) and n >= 0 for n in entity_counts)
        assert train_entity_counts is None or util.all_and_any(n in entity_counts for n in train_entity_counts)
        assert validation_entity_counts is None or util.all_and_any(n in entity_counts for n in validation_entity_counts)
        assert test_entity_counts is None or util.all_and_any(n in entity_counts for n in test_entity_counts)
        self.entity_counts = entity_counts
        self.train_entity_counts = util.value_or_default(train_entity_counts, entity_counts)
        self.validation_entity_counts = util.value_or_default(validation_entity_counts, entity_counts)
        self.validation_count_rate = validation_count_rate
        self.test_entity_counts = util.value_or_default(test_entity_counts, entity_counts)
        self.test_count_rate = test_count_rate

        assert (validation_combinations is None) == (test_combinations is None)
        self.validation_combinations = validation_combinations
        self.validation_space_rate_range = validation_space_rate_range
        self.validation_combination_rate = validation_combination_rate
        self.test_combinations = test_combinations
        self.test_space_rate_range = test_space_rate_range
        self.test_combination_rate = test_combination_rate
        self.invalid_combinations = set()
        self.invalid_validation_combinations = set()

        if self.validation_combinations is not None:
            assert len(self.validation_combinations) > 0
            validation_shapes = set(shape for shape, _, _ in self.validation_combinations)
            validation_colors = set(color for _, color, _ in self.validation_combinations)
            validation_textures = set(texture for _, _, texture in self.validation_combinations)
            self.validation_space = [(shape, color, texture) for shape in validation_shapes for color in validation_colors for texture in validation_textures]
            self.invalid_combinations.update(self.validation_combinations)

        if self.test_combinations is not None:
            assert len(self.test_combinations) > 0
            test_shapes = set(shape for shape, _, _ in self.test_combinations)
            test_colors = set(color for _, color, _ in self.test_combinations)
            test_textures = set(texture for _, _, texture in self.test_combinations)
            self.test_space = [(shape, color, texture) for shape in test_shapes for color in test_colors for texture in test_textures]
            self.invalid_combinations.update(self.test_combinations)
            self.invalid_validation_combinations.update(self.test_combinations)

    def initialize(self, mode):
        super(GenericGenerator, self).initialize(mode=mode)
        if mode is None:
            self.num_entities = choice(self.entity_counts)
        elif mode == 'train':
            self.num_entities = choice(self.train_entity_counts)
        elif mode == 'validation':
            if random() < self.validation_count_rate:
                self.num_entities = choice(self.validation_entity_counts)
            else:
                self.num_entities = choice(self.train_entity_counts)
        elif mode == 'test':
            if random() < self.test_count_rate:
                self.num_entities = choice(self.test_entity_counts)
            else:
                self.num_entities = choice(self.train_entity_counts)
        self.validation_space_rate = uniform(*self.validation_space_rate_range)
        self.test_space_rate = uniform(*self.test_space_rate_range)

    def model(self):
        return util.merge_dicts(
            dict1=super(GenericGenerator, self).model(),
            dict2=dict(
                num_entities=self.num_entities,
                validation_space_rate=self.validation_space_rate,
                test_space_rate=self.test_space_rate
            )
        )

    def sample_entity(self, world, last_entity, combinations=None):
        raise NotImplementedError

    def generate_world(self):
        world = World(self.world_size, self.world_color)
        n = 0
        last_entity = -1

        if self.num_entities == 0:
            return world

        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            entity = self.sample_entity(world=world, last_entity=last_entity)
            if world.add_entity(entity, collision_tolerance=self.collision_tolerance, collision_shade_difference=self.collision_shade_difference, boundary_tolerance=self.boundary_tolerance):
                last_entity = entity
                n += 1
                if n == self.num_entities:
                    break
            else:
                last_entity = None
        else:
            return None

        if self.collision_tolerance:
            world.sort_entities()

        return world

    def generate_train_world(self):
        world = World(self.world_size, self.world_color)
        n = 0
        last_entity = -1

        if self.num_entities == 0:
            return world

        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            entity = self.sample_entity(world=world, last_entity=last_entity)
            combination = (entity.shape.name, entity.color.name, entity.texture.name)
            if combination in self.invalid_combinations:
                last_entity = None
            elif world.add_entity(entity, collision_tolerance=self.collision_tolerance, collision_shade_difference=self.collision_shade_difference, boundary_tolerance=self.boundary_tolerance):
                last_entity = entity
                n += 1
                if n == self.num_entities:
                    break
            else:
                last_entity = None
        else:
            return None

        if self.collision_tolerance:
            world.sort_entities()

        return world

    def generate_validation_world(self):
        if self.validation_combinations is None:
            return self.generate_train_world()

        world = World(self.world_size, self.world_color)
        n = 0
        last_entity = -1

        if self.num_entities == 0:
            return world

        if self.validation_combination_rate is not None:
            while True:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.validation_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance, collision_shade_difference=self.collision_shade_difference):
                    n += 1
                    last_entity = entity
                    break
                else:
                    last_entity = None

        if self.num_entities == 1:
            return world

        pick_space = random() < self.validation_space_rate
        pick_combination = pick_space and random() < self.validation_combination_rate
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            if pick_combination:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.validation_combinations)
            elif pick_space:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.validation_space)
            else:
                entity = self.sample_entity(world=world, last_entity=last_entity)
            combination = (entity.shape.name, entity.color.name, entity.texture.name)
            if combination in self.invalid_validation_combinations:
                last_entity = None
            elif world.add_entity(entity, collision_tolerance=self.collision_tolerance, collision_shade_difference=self.collision_shade_difference, boundary_tolerance=self.boundary_tolerance):
                n += 1
                if n == self.num_entities:
                    break
                last_entity = entity
                pick_space = random() < self.validation_space_rate
                pick_combination = pick_space and random() < self.validation_combination_rate
            else:
                last_entity = None
        else:
            return None

        if self.collision_tolerance:
            world.sort_entities()

        return world

    def generate_test_world(self):
        if self.test_combinations is None:
            return self.generate_validation_world()

        world = World(self.world_size, self.world_color)
        n = 0
        last_entity = -1

        if self.num_entities == 0:
            return world

        if self.test_combination_rate is not None:
            while True:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.test_combinations)
                if world.add_entity(entity, boundary_tolerance=self.boundary_tolerance, collision_tolerance=self.collision_tolerance, collision_shade_difference=self.collision_shade_difference):
                    n += 1
                    last_entity = entity
                    break
                else:
                    last_entity = None

        if self.num_entities == 1:
            return world

        pick_space = random() < self.test_space_rate
        pick_combination = pick_space and random() < self.test_combination_rate
        for _ in range(self.num_entities * self.__class__.MAX_ATTEMPTS):
            if pick_combination:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.test_combinations)
            elif pick_space:
                entity = self.sample_entity(world=world, last_entity=last_entity, combinations=self.test_space)
            else:
                entity = self.sample_entity(world=world, last_entity=last_entity)
            if world.add_entity(entity, collision_tolerance=self.collision_tolerance, collision_shade_difference=self.collision_shade_difference, boundary_tolerance=self.boundary_tolerance):
                n += 1
                if n == self.num_entities:
                    break
                last_entity = entity
                pick_space = random() < self.test_space_rate
                pick_combination = pick_space and random() < self.test_combination_rate
            else:
                last_entity = None
        else:
            return None

        if self.collision_tolerance:
            world.sort_entities()

        return world
