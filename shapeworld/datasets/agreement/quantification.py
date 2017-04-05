from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import QuantificationCaptioner


class QuantificationDataset(CaptionAgreementDataset):

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, shapes_range, colors_range, textures_range, caption_size, words, caption_modes=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, noise_range=None, collision_tolerance=None, boundary_tolerance=None, realizer=None, quantifier_tolerance=None, **kwargs):
        world_generator = GenericGenerator(entity_counts, world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, noise_range, collision_tolerance, boundary_tolerance, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts, shapes_range=shapes_range, colors_range=colors_range, textures_range=textures_range)
        world_captioner = QuantificationCaptioner(world_generator.shapes, world_generator.colors, world_generator.textures, realizer=realizer, quantifier_tolerance=quantifier_tolerance, modes=caption_modes)
        super().__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            incorrect_world_ratio=0.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio)


dataset = QuantificationDataset
QuantificationDataset.dataset_name = 'quantification'
QuantificationDataset.default_config = {
    'entity_counts': [3, 4, 5, 6, 7, 8],
    'train_entity_counts': [3, 4, 5, 7],
    'validation_entity_counts': [6],
    'test_entity_counts': [8],
    'shapes_range': [2, 4],
    'colors_range': [2, 4],
    'textures_range': [1, 1],
    'caption_size': 7,
    'words': ['a', 'all', 'an', 'are', 'black', 'blue', 'circle', 'circles', 'cross', 'crosses', 'cyan', 'ellipse', 'ellipses', 'every', 'green', 'is', 'magenta', 'most', 'no', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'semicircle', 'semicircles', 'shape', 'shapes', 'some', 'square', 'squares', 'the', 'triangle', 'triangles', 'two', 'white', 'yellow', '.']
}
