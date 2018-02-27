from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import RandomAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, ExistentialCaptioner


class Multishape(CaptionAgreementDataset):

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
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=0.25,
        entity_counts=(1, 2, 3, 4, 5, 6, 7, 8),
        train_entity_counts=(1, 2, 3, 4, 5, 7),
        validation_entity_counts=(6,),
        test_entity_counts=(8,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        max_provoke_collision_rate=0.33,
        caption_size=8,
        vocabulary=('.', 'a', 'an', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'gray', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'yellow'),
        correct_ratio=0.5,
        train_correct_ratio=None,
        validation_correct_ratio=None,
        test_correct_ratio=None,
        worlds_per_instance=1,
        captions_per_instance=1,
        pixel_noise_stddev=0.0,
        caption_realizer='dmrs',
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
            collision_tolerance=collision_tolerance,
            collision_shade_difference=collision_shade_difference,
            boundary_tolerance=boundary_tolerance,
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=max_provoke_collision_rate
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
                                RegularTypeCaptioner(
                                    hypernym_rate=0.0
                                )
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
            correct_ratio=correct_ratio,
            train_correct_ratio=train_correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio,
            worlds_per_instance=worlds_per_instance,
            captions_per_instance=captions_per_instance,
            pixel_noise_stddev=pixel_noise_stddev,
            caption_realizer=caption_realizer,
            language=language
        )


dataset = Multishape
