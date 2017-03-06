from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators.generic import GenericWorldGenerator
from shapeworld.captioners.dmrs.existential import ExistentialDmrsCaptioner


class OneShapeDataset(CaptionAgreementDataset):

    def __init__(self, world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, validation_shapes, validation_textures, validation_colors, test_shapes, test_textures, test_colors, train_combinations, caption_size, words, correct_ratio, train_correct_ratio, validation_correct_ratio, test_correct_ratio, **kwargs):
        super().__init__(
            world_generator=GenericWorldGenerator(world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, [1], validation_shapes=validation_shapes, validation_textures=validation_textures, validation_colors=validation_colors, test_shapes=test_shapes, test_textures=test_textures, test_colors=test_colors, train_combinations=train_combinations),
            world_captioner=ExistentialDmrsCaptioner(caption_size, words),
            incorrect_world_ratio=1.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio)


dataset = OneShapeDataset
OneShapeDataset.name = 'OneShape'
OneShapeDataset.default_config = {
    'world_size': 64,
    'world_color': 'black',
    'noise_range': 0.1,
    'shapes': ['square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'],
    'size_range': [0.2, 0.3],
    'distortion_range': [2.0, 3.0],
    'colors': ['black', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white'],
    'shade_range': 0.5,
    'textures': ['solid'],
    'rotation': True,
    'validation_shapes': ['square'],
    'validation_textures': ['solid'],
    'validation_colors': ['red'],
    'test_shapes': ['square'],
    'test_textures': ['solid'],
    'test_colors': ['red'],
    'train_combinations': None,
    'caption_size': 6,
    'words': ['a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'square', 'there', 'triangle', 'white', 'yellow', '.'],
    'correct_ratio': 0.5,
    'train_correct_ratio': 0.33,
    'validation_correct_ratio': 0.5,
    'test_correct_ratio': 0.5}
