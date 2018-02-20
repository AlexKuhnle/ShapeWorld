from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, ExistentialCaptioner


class Oneshape(CaptionAgreementDataset):

    def __init__(
        self,
        world_size=64,
        world_color='black',
        shapes=None,
        colors=None,
        textures=None,
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        boundary_tolerance=0.25,
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=9,
        vocabulary=('.', 'a', 'an', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow'),
        correct_ratio=0.5,
        train_correct_ratio=0.5,
        validation_correct_ratio=0.5,
        test_correct_ratio=0.5,
        worlds_per_instance=1,
        captions_per_instance=1,
        caption_realizer=None,
        language=None
    ):

        world_generator = RandomAttributesGenerator(
            world_size=world_size,
            world_color=world_color,
            shapes=shapes,
            colors=colors,
            textures=textures,
            rotation=rotation,
            size_range=size_range,
            distortion_range=distortion_range,
            shade_range=shade_range,
            boundary_tolerance=boundary_tolerance,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
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
                                    hypernym_rate=0.0,
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
            correct_ratio=correct_ratio,
            train_correct_ratio=train_correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio,
            worlds_per_instance=worlds_per_instance,
            captions_per_instance=captions_per_instance,
            caption_realizer=caption_realizer,
            language=language
        )


dataset = Oneshape
