from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, ExistentialCaptioner


class Oneshape(CaptionAgreementDataset):

    def __init__(
        self,
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=9,
        vocabulary=('.', 'a', 'an', 'angular', 'asymmetric', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'round', 'semicircle', 'shape', 'square', 'symmetric', 'there', 'triangle', 'yellow'),
        language=None
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=[1],
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )

        world_captioner = CaptionerMixer(
            captioners=(
                RegularTypeCaptioner(
                    existing_attribute_rate=0.0
                ),
                ExistentialCaptioner(
                    restrictor_captioner=RegularTypeCaptioner(
                        hypernym_rate=1.0,
                        existing_attribute_rate=0.0,
                        logical_tautology_rate=1.0
                    ),
                    body_captioner=AttributeTypeRelationCaptioner(
                        attribute_type_captioner=CaptionerMixer(
                            captioners=(
                                RegularAttributeCaptioner(
                                    existing_attribute_rate=0.0
                                ),
                                RegularTypeCaptioner(
                                    existing_attribute_rate=0.0
                                )
                            )
                        )
                    ),
                    pragmatical_tautology_rate=1.0
                )
            )
        )

        super(Oneshape, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = Oneshape
