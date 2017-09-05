from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import AttributesTypeCaptioner, AttributesRelationCaptioner, AbsoluteQuantifierCaptioner


class CountingSimpleDataset(CaptionAgreementDataset):

    dataset_name = 'counting_simple'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, validation_combinations, test_combinations, shapes_range, colors_range, textures_range, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=entity_counts,
            collision_tolerance=0.0,
            boundary_tolerance=0.0,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            shapes_range=shapes_range,
            colors_range=colors_range,
            textures_range=textures_range,
            max_provoke_collision_rate=0.0
        )
        world_captioner = AbsoluteQuantifierCaptioner(
            restrictor_captioner=AttributesTypeCaptioner(
                hypernym_ratio=1.0
            ),
            body_captioner=AttributesRelationCaptioner()
        )
        super(CountingSimpleDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = CountingSimpleDataset
CountingSimpleDataset.default_config = dict(
    entity_counts=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    train_entity_counts=[5, 6, 7, 8, 9, 10, 11, 12, 14],
    validation_entity_counts=[13],
    test_entity_counts=[15],
    validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    shapes_range=[2, 3],
    colors_range=[2, 3],
    textures_range=[1, 1],
    caption_size=9,
    words=['.', 'a', 'an', 'are', 'black', 'blue', 'both', 'circle', 'circles', 'cross', 'crosses', 'cyan', 'ellipse', 'ellipses', 'exactly', 'five', 'four', 'green', 'is', 'magenta', 'one', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'semicircle', 'semicircles', 'shape', 'shapes', 'square', 'squares', 'the', 'three', 'triangle', 'triangles', 'two', 'white', 'yellow']
)
