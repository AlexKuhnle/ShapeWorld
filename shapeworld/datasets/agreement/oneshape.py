from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesTypeCaptioner, AttributesRelationCaptioner, ExistentialCaptioner


class OneshapeDataset(CaptionAgreementDataset):

    dataset_name = 'oneshape'

    def __init__(self, validation_combinations, test_combinations, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=[1]
        )
        world_captioner = CaptionerMixer(
            captioners=(
                AttributesTypeCaptioner(
                    existing_attribute_ratio=0.0
                ),
                ExistentialCaptioner(
                    restrictor_captioner=AttributesTypeCaptioner(
                        hypernym_ratio=1.0,
                        existing_attribute_ratio=0.0
                    ),
                    body_captioner=AttributesRelationCaptioner(
                        existing_attribute_ratio=0.0
                    )
                )
            ),
            trivial_acceptance_rate=1.0
        )
        super(OneshapeDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = OneshapeDataset
OneshapeDataset.default_config = dict(
    validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    caption_size=8,
    words=['.', 'a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'white', 'yellow']
)
