from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import AttributesTypeCaptioner


class OneshapeSimpleDataset(CaptionAgreementDataset):

    dataset_name = 'oneshape_simple'

    def __init__(self, validation_combinations, test_combinations, caption_size, words, language=None):
        world_generator = GenericGenerator(
            entity_counts=[1],
            collision_tolerance=0.0,
            boundary_tolerance=0.0
        )
        world_captioner = AttributesTypeCaptioner(
            existing_attribute_ratio=0.0,
            trivial_acceptance_rate=1.0
        )
        super(OneshapeSimpleDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            language=language
        )


dataset = OneshapeSimpleDataset
OneshapeSimpleDataset.default_config = dict(
    validation_combinations=[['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    test_combinations=[['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    caption_size=6,
    words=['.', 'a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'white', 'yellow']
)
