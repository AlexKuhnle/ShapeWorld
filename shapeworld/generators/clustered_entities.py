from math import sqrt
from random import choice, randint, random
from shapeworld.util import Point
from shapeworld.world import Entity
from shapeworld.generators.generic2 import GenericGenerator


class ClusteredEntitiesGenerator(GenericGenerator):

    MAX_NUM_CLUSTERS = 5
    MIN_CLUSTERS_DISTANCE = 0.4
    MAX_CLUSTER_CENTER_DISTANCE = 0.25

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, validation_combinations, test_combinations, clusters_range, **kwargs):
        super(ClusteredEntitiesGenerator, self).__init__(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            **kwargs
        )
        assert 1 <= clusters_range[0] <= clusters_range[1] <= self.__class__.MAX_NUM_CLUSTERS
        self.clusters_range = clusters_range
        self.clusters_range = (3, 3)

    def initialize(self, mode):
        super(ClusteredEntitiesGenerator, self).initialize(mode=mode)
        self.num_clusters = randint(*self.clusters_range)

    def sample_entity(self, world, last_entity, combinations=None):
        if last_entity == -1:
            while True:
                self.clusters = list()
                for _ in range(self.num_clusters * self.__class__.MAX_ATTEMPTS):
                    r = random()
                    if r < 0.25:
                        cluster_center = Point(0.125 + r * 3.0, 0.15)
                    elif r < 0.5:
                        cluster_center = Point(0.125 + (r - 0.25) * 3.0, 0.85)
                    elif r < 0.75:
                        cluster_center = Point(0.15, 0.125 + (r - 0.5) * 3.0)
                    else:
                        cluster_center = Point(0.85, 0.125 + (r - 0.75) * 3.0)
                    # cluster_center = world.random_location()
                    if all(cluster_center.distance(cluster) > self.__class__.MIN_CLUSTERS_DISTANCE for cluster in self.clusters):
                        self.clusters.append(cluster_center)
                        if len(self.clusters) == self.num_clusters:
                            break
                else:
                    print(self.clusters)
                    continue
                break
        elif last_entity is not None:
            pass
        self.selected_cluster = choice(self.clusters)
        angle = Point.from_angle(angle=random())
        center = self.selected_cluster + angle * random() * self.__class__.MAX_CLUSTER_CENTER_DISTANCE
        if combinations is None:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, shapes=self.shapes, colors=self.colors, textures=self.textures)
        else:
            return Entity.random_instance(center=center, rotation=self.rotation, size_range=self.size_range, distortion_range=self.distortion_range, shade_range=self.shade_range, combinations=combinations)
