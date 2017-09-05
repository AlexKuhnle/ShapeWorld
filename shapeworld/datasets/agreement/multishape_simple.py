from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import AttributesTypeCaptioner


class MultishapeSimpleDataset(CaptionAgreementDataset):

    dataset_name = 'multishape_simple'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, validation_combinations, test_combinations, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=entity_counts,
            collision_tolerance=0.0,
            boundary_tolerance=0.0,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=0.0
        )
        world_captioner = AttributesTypeCaptioner()
        super(MultishapeSimpleDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = MultishapeSimpleDataset
MultishapeSimpleDataset.default_config = dict(
    entity_counts=[2, 3, 4, 5, 6, 7, 8],
    train_entity_counts=[2, 3, 4, 5, 7],
    validation_entity_counts=[6],
    test_entity_counts=[8],
    validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    caption_size=6,
    words=['.', 'a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'white', 'yellow']
)
