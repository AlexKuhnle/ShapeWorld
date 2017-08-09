from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesNounCaptioner, AttributesRelationCaptioner, SpatialRelationCaptioner, ComparisonRelationCaptioner, AbsoluteQuantifierCaptioner


class CountingDataset(CaptionAgreementDataset):

    dataset_name = 'counting'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, shapes_range, colors_range, textures_range, caption_size, words, incorrect_caption_distribution=None, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, realizer=None, language=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, quantifier_tolerance=None, **kwargs):
        world_generator = GenericGenerator(entity_counts, world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts, shapes_range=shapes_range, colors_range=colors_range, textures_range=textures_range)
        body_captioner = CaptionerMixer(
            captioners=(
                AttributesRelationCaptioner(),
                SpatialRelationCaptioner(),
                ComparisonRelationCaptioner()
            ),
            distribution=distribution,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            test_distribution=test_distribution
        )
        world_captioner = AbsoluteQuantifierCaptioner(
            restrictor_captioner=AttributesNounCaptioner(),
            body_captioner=body_captioner,
            incorrect_distribution=incorrect_caption_distribution
        )
        super(CountingDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            incorrect_world_ratio=0.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio,
            caption_realizer=realizer,
            realizer_language=language
        )


dataset = CountingDataset
CountingDataset.default_config = {
    'entity_counts': [3, 4, 5, 6, 7, 8],
    'train_entity_counts': [3, 4, 5, 7],
    'validation_entity_counts': [6],
    'test_entity_counts': [8],
    'validation_combinations': [['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    'test_combinations': [['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    'shapes_range': [2, 4],
    'colors_range': [2, 4],
    'textures_range': [1, 1],
    'caption_size': 16,
    'words': ['.', 'a', 'above', 'an', 'are', 'below', 'bigger', 'black', 'blue', 'both', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'ellipse', 'ellipses', 'exactly', 'farther', 'farthest', 'five', 'four', 'from', 'green', 'is', 'left', 'lighter', 'magenta', 'of', 'one', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'right', 'semicircle', 'semicircles', 'shape', 'shapes', 'smaller', 'square', 'squares', 'than', 'the', 'three', 'to', 'triangle', 'triangles', 'two', 'white', 'yellow']
}
