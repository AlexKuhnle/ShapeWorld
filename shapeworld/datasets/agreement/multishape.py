from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, ExistentialCaptioner


class Multishape(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(2, 3, 4, 5, 6, 7, 8),
        train_entity_counts=(2, 3, 4, 5, 7),
        validation_entity_counts=(6,),
        test_entity_counts=(8,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=8,
        vocabulary=('.', 'a', 'an', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow'),
        language=None
    ):

        world_generator = RandomAttributesGenerator(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )

        world_captioner = CaptionerMixer(
            captioners=(
                RegularTypeCaptioner(),
                ExistentialCaptioner(
                    restrictor_captioner=RegularTypeCaptioner(
                        hypernym_rate=1.0,
                        logical_tautology_rate=1.0
                    ),
                    body_captioner=AttributeTypeRelationCaptioner(
                        attribute_type_captioner=CaptionerMixer(
                            captioners=(
                                RegularAttributeCaptioner(),
                                RegularTypeCaptioner()
                            )
                        )
                    )
                )
            )
        )

        super(Multishape, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = Multishape
