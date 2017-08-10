from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesNounCaptioner, AttributesRelationCaptioner, SpatialRelationCaptioner, ComparisonRelationCaptioner, ExistentialCaptioner, AbsoluteQuantifierCaptioner, RelativeQuantifierCaptioner, ConjunctionCaptioner, DisjunctionCaptioner


class CombinationDataset(CaptionAgreementDataset):

    dataset_name = 'combination'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, shapes_range, colors_range, textures_range, caption_size, words, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, realizer=None, language=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, quantifier_tolerance=None, **kwargs):
        world_generator = GenericGenerator(entity_counts, world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts, shapes_range=shapes_range, colors_range=colors_range, textures_range=textures_range)
        oneshape = CaptionerMixer(
            captioners=(
                AttributesNounCaptioner(),
                ExistentialCaptioner(
                    subject_captioner=AttributesNounCaptioner(hypernym_ratio=1.0),
                    verb_captioner=AttributesRelationCaptioner()
                )
            )
        )
        relational = CaptionerMixer(
            captioners=(
                SpatialRelationCaptioner(),
                ComparisonRelationCaptioner()
            )
        )
        counting = AbsoluteQuantifierCaptioner(
            restrictor_captioner=AttributesNounCaptioner(),
            body_captioner=CaptionerMixer(
                captioners=(
                    AttributesRelationCaptioner(),
                    SpatialRelationCaptioner(),
                    ComparisonRelationCaptioner()
                )
            )
        )
        quantification = RelativeQuantifierCaptioner(
            restrictor_captioner=AttributesNounCaptioner(),
            body_captioner=CaptionerMixer(
                captioners=(
                    AttributesRelationCaptioner(),
                    SpatialRelationCaptioner(),
                    ComparisonRelationCaptioner()
                )
            )
        )
        world_captioner = CaptionerMixer(
            captioners=(
                ConjunctionCaptioner(captioners=(oneshape, relational, counting, quantification)),
                DisjunctionCaptioner(captioners=(oneshape, relational, counting, quantification))
            ),
            distribution=distribution,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            test_distribution=test_distribution
        )
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
    'entity_counts': [1, 2, 3, 4, 5, 6, 7, 8],
    'train_entity_counts': [1, 2, 3, 4, 5, 7],
    'validation_entity_counts': [6],
    'test_entity_counts': [8],
    'validation_combinations': [['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    'test_combinations': [['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    'shapes_range': [2, 4],
    'colors_range': [2, 4],
    'textures_range': [1, 1],
    'caption_size': 28,
    'words': ['.', 'a', 'above', 'all', 'an', 'and', 'are', 'behind', 'below', 'bigger', 'black', 'blue', 'both', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'either', 'ellipse', 'ellipses', 'every', 'exactly', 'farther', 'farthest', 'five', 'four', 'from', 'front', 'green', 'half', 'in', 'is', 'left', 'lighter', 'magenta', 'most', 'no', 'of', 'one', 'or', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'right', 'semicircle', 'semicircles', 'shape', 'shapes', 'smaller', 'some', 'square', 'squares', 'than', 'the', 'there', 'three', 'to', 'triangle', 'triangles', 'two', 'white', 'yellow']
}
