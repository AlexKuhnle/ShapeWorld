from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import RegularTypeCaptioner, RelationCaptioner, ExistentialCaptioner


class SpatialSimple(CaptionAgreementDataset):

    def __init__(
        self,
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=12,
        vocabulary=('.', 'a', 'above', 'an', 'below', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'left', 'magenta', 'of', 'pentagon', 'rectangle', 'red', 'right', 'semicircle', 'shape', 'square', 'the', 'to', 'triangle', 'yellow'),
        language=None
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=[2],
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=0.0,
            collision_tolerance=0.0,
            boundary_tolerance=0.0
        )

        world_captioner = ExistentialCaptioner(
            restrictor_captioner=RegularTypeCaptioner(
            ),
            body_captioner=RelationCaptioner(
                reference_captioner=RegularTypeCaptioner(),
                comparison_captioner=RegularTypeCaptioner(),
                relations=('x-rel', 'y-rel')
            ),
            pragmatical_tautology_rate=1.0
        )

        super(SpatialSimple, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = SpatialSimple
