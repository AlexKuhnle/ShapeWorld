from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesNounCaptioner, SpatialRelationCaptioner, ComparisonRelationCaptioner, ExistentialCaptioner


class CombinationDataset(CaptionAgreementDataset):

    dataset_name = 'relational'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, caption_size, words, incorrect_caption_distribution=None, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, realizer=None, language=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, quantifier_tolerance=None, **kwargs):
        world_generator = GenericGenerator(entity_counts, world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts)
        verb_captioner = CaptionerMixer(
            captioners=(
                SpatialRelationCaptioner(),
                ComparisonRelationCaptioner()),
            distribution=distribution,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            test_distribution=test_distribution
        )
        world_captioner = ExistentialCaptioner(subject_captioner=AttributesNounCaptioner(), verb_captioner=verb_captioner, incorrect_distribution=incorrect_caption_distribution)
        super(CombinationDataset, self).__init__(
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
            language=language
        )


dataset = CombinationDataset
CombinationDataset.default_config = {
    'entity_counts': [3, 4, 5, 6, 7, 8],
    'train_entity_counts': [3, 4, 5, 7],
    'validation_entity_counts': [6],
    'test_entity_counts': [8],
    'validation_combinations': [['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    'test_combinations': [['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    'caption_size': 14,
    'words': ['.', 'a', 'above', 'an', 'below', 'bigger', 'black', 'blue', 'circle', 'closer', 'closest', 'cross', 'cyan', 'darker', 'ellipse', 'farther', 'farthest', 'from', 'green', 'is', 'left', 'lighter', 'magenta', 'of', 'pentagon', 'rectangle', 'red', 'right', 'semicircle', 'shape', 'smaller', 'square', 'than', 'the', 'to', 'triangle', 'white', 'yellow']
}
