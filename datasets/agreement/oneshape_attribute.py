from random import random
from shapeworld.shape import Shape
from shapeworld.color import Color, Fill
from shapeworld.world import Entity, World
from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators.generic import GenericWorldGenerator
from shapeworld.captioners.dmrs.existential import ExistentialDmrsCaptioner


class OneShapeAttributeDataset(CaptionAgreementDataset):

    def __init__(self, world_size, shapes, size_range, distortion_range, rotation, fills, colors, shade_range, noise_range, world_background_color, validation_shapes, validation_fills, validation_colors, test_shapes, test_fills, test_colors, train_combinations, caption_size, words, correct_ratio, train_correct_ratio, validation_correct_ratio, test_correct_ratio, **kwargs):
        super().__init__(
            world_generator=GenericWorldGenerator(world_size, shapes, size_range, distortion_range, rotation, fills, colors, shade_range, noise_range, world_background_color, [1], validation_shapes=validation_shapes, validation_fills=validation_fills, validation_colors=validation_colors, test_shapes=test_shapes, test_fills=test_fills, test_colors=test_colors, train_combinations=train_combinations),
            world_captioner=ExistentialDmrsCaptioner(caption_size, words),
            incorrect_world_ratio=1.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio)

    def generate_incorrect_world(self, world, caption, mode):
        if mode != 'train':
            mode = None
        while True:
            if random() < 0.5:
                shapes = [value for shape, value in self.world_generator.shapes.items() if shape != str(world.entities[0].shape)]
                fills = [Fill.fills[str(world.entities[0].fill)]]
                colors = [Color.colors[str(world.entities[0].fill.color)]]
            else:
                shapes = [Shape.shapes[str(world.entities[0].shape)]]
                fills = [Fill.fills[str(world.entities[0].fill)]]
                colors = [value for color, value in self.world_generator.colors.items() if color != str(world.entities[0].fill.color)]
            world = World(world_size=self.world_generator.world_size, background=self.world_generator.world_background, noise_range=self.world_generator.noise_range)
            n = 0
            while n < 1:
                entity = Entity.random_instance(center=world.random_center(), shapes=shapes, size_range=self.world_generator.size_range, distortion_range=self.world_generator.distortion_range, rotation=self.world_generator.rotation, fills=fills, colors=colors, shade_range=self.world_generator.shade_range)
                n += world.add_entity(entity)
            if caption.agreement(world) == 0.0:
                return world


dataset = OneShapeAttributeDataset
OneShapeAttributeDataset.default_config = {
    'world_size': [64, 64],
    'shapes': ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'],
    'size_range': [0.1, 0.15],
    'distortion_range': [2.0, 3.0],
    'rotation': True,
    'fills': ['solid'],
    'colors': ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white'],
    'shade_range': 0.5,
    'noise_range': 0.1,
    'world_background_color': 'black',
    'validation_shapes': ['square'],
    'validation_fills': ['solid'],
    'validation_colors': ['red'],
    'test_shapes': ['square'],
    'test_fills': ['solid'],
    'test_colors': ['red'],
    'train_combinations': None,
    'caption_size': 5,
    'words': ['a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'square', 'there', 'triangle', 'white', 'yellow', '.'],
    'correct_ratio': 0.5,
    'train_correct_ratio': 0.33,
    'validation_correct_ratio': 0.5,
    'test_correct_ratio': 0.5}
