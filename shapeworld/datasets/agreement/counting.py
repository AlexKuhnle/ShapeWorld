from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesTypeCaptioner, AttributesRelationCaptioner, SpatialRelationCaptioner, ComparisonRelationCaptioner, AbsoluteQuantifierCaptioner


class CountingDataset(CaptionAgreementDataset):

    dataset_name = 'counting'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, validation_combinations, test_combinations, shapes_range, colors_range, textures_range, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            shapes_range=shapes_range,
            colors_range=colors_range,
            textures_range=textures_range
        )
        body_captioner = CaptionerMixer(
            captioners=(
                AttributesRelationCaptioner(),
                SpatialRelationCaptioner(relations=('x-rel', 'y-rel', 'proximity-rel')),
                ComparisonRelationCaptioner()
            )
        )
        world_captioner = AbsoluteQuantifierCaptioner(
            restrictor_captioner=AttributesTypeCaptioner(),
            body_captioner=body_captioner
        )
        super(CountingDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = CountingDataset
CountingDataset.default_config = dict(
    entity_counts=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    train_entity_counts=[5, 6, 7, 8, 9, 10, 11, 12, 14],
    validation_entity_counts=[13],
    test_entity_counts=[15],
    validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    shapes_range=[2, 3],
    colors_range=[2, 3],
    textures_range=[1, 1],
    caption_size=16,
    words=['.', 'a', 'above', 'an', 'are', 'behind', 'below', 'bigger', 'black', 'blue', 'both', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'ellipse', 'ellipses', 'exactly', 'farther', 'farthest', 'five', 'four', 'from', 'front', 'green', 'in', 'is', 'left', 'lighter', 'magenta', 'of', 'one', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'right', 'semicircle', 'semicircles', 'shape', 'shapes', 'smaller', 'square', 'squares', 'than', 'the', 'three', 'to', 'triangle', 'triangles', 'two', 'white', 'yellow']
)
