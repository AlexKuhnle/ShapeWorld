from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesTypeCaptioner, AttributesRelationCaptioner, ExistentialCaptioner


class MultishapeDataset(CaptionAgreementDataset):

    dataset_name = 'multishape'

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, validation_combinations, test_combinations, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )
        world_captioner = CaptionerMixer(
            captioners=(
                AttributesTypeCaptioner(),
                ExistentialCaptioner(
                    restrictor_captioner=AttributesTypeCaptioner(
                        hypernym_ratio=1.0
                    ),
                    body_captioner=AttributesRelationCaptioner()
                )
            )
        )
        super(MultishapeDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = MultishapeDataset
MultishapeDataset.default_config = dict(
    entity_counts=[2, 3, 4, 5, 6, 7, 8],
    train_entity_counts=[2, 3, 4, 5, 7],
    validation_entity_counts=[6],
    test_entity_counts=[8],
    validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    caption_size=8,
    words=['.', 'a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'white', 'yellow']
)
