from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators.generic import GenericWorldGenerator
from shapeworld.captioners.dmrs.quantifier import QuantifierDmrsCaptioner


class QuantifierDataset(CaptionAgreementDataset):

    def __init__(self, world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, shapes_range, colors_range, caption_size, words, correct_ratio, train_correct_ratio, validation_correct_ratio, test_correct_ratio, **kwargs):
        world_generator = GenericWorldGenerator(world_size, world_color, noise_range, shapes, size_range, distortion_range, colors, shade_range, textures, rotation, entity_counts, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts, shapes_range=shapes_range, colors_range=colors_range)
        world_captioner = QuantifierDmrsCaptioner(caption_size, words, world_generator.shapes, world_generator.colors)
        super().__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            incorrect_world_ratio=1.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio)


dataset = QuantifierDataset
QuantifierDataset.name = 'Quantifier'
QuantifierDataset.default_config = {
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
    'entity_counts': [3, 4, 5, 6, 7, 8],
    'train_entity_counts': [3, 4, 5, 6],
    'validation_entity_counts': [7],
    'test_entity_counts': [8],
    'shapes_range': [2, 4],
    'colors_range': [2, 4],
    'caption_size': 7,
    'words': ['a', 'all', 'an', 'are', 'black', 'blue', 'circle', 'circles', 'cross', 'crosses', 'cyan', 'ellipse', 'ellipses', 'every', 'green', 'is', 'magenta', 'most', 'no', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'semicircle', 'semicircles', 'shape', 'shapes', 'some', 'square', 'squares', 'the', 'triangle', 'triangles', 'two', 'white', 'yellow', '.'],
    'correct_ratio': 0.5,
    'train_correct_ratio': 0.33,
    'validation_correct_ratio': 0.5,
    'test_correct_ratio': 0.5
}
