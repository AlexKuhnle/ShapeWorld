from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GeneratorMixer, RandomAttributesGenerator, ReinforcedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, RelationCaptioner, ExistentialCaptioner, QuantifierCaptioner, NumberBoundCaptioner, ComparativeQuantifierCaptioner, ConjunctionCaptioner, DisjunctionCaptioner


class Combination(CaptionAgreementDataset):

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
        entity_counts=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(1, 2, 4, 6, 7, 9, 11, 12, 14),
        validation_entity_counts=(3, 8, 13),
        test_entity_counts=(5, 10, 15),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        max_provoke_collision_rate=0.33,
        reinforcement_range=(1, 3),
        caption_size=28,
        vocabulary=('.', 'a', 'above', 'all', 'almost', 'an', 'and', 'are', 'as', 'at', 'behind', 'below', 'bigger', 'blue', 'but', 'circle', 'circles', 'closer', 'cross', 'crosses', 'cyan', 'darker', 'eight', 'either', 'ellipse', 'ellipses', 'exactly', 'farther', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'in', 'is', 'least', 'left', 'less', 'lighter', 'magenta', 'many', 'more', 'most', 'no', 'none', 'not', 'of', 'one', 'or', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red', 'right', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'square', 'squares', 'than', 'the', 'there', 'third', 'thirds', 'three', 'to', 'triangle', 'triangles', 'twice', 'two', 'yellow', 'zero'),
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

        random_generator = RandomAttributesGenerator(
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
        reinforced_attributes_generator = ReinforcedAttributesGenerator(
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
            max_provoke_collision_rate=max_provoke_collision_rate,
            reinforcement_range=reinforcement_range
        )
        world_generator = GeneratorMixer(
            generators=(random_generator, reinforced_attributes_generator)
        )

        body_captioner = AttributeTypeRelationCaptioner(
            attribute_type_captioner=CaptionerMixer(
                captioners=(
                    RegularAttributeCaptioner(),
                    RegularTypeCaptioner()
                )
            )
        )
        existential_captioner = CaptionerMixer(
            captioners=(
                RegularTypeCaptioner(),
                ExistentialCaptioner(
                    restrictor_captioner=RegularTypeCaptioner(
                        hypernym_rate=1.0,
                        logical_tautology_rate=1.0
                    ),
                    body_captioner=body_captioner
                )
            )
        )
        relation_captioner = ExistentialCaptioner(
            restrictor_captioner=RegularTypeCaptioner(),
            body_captioner=RelationCaptioner(
                reference_captioner=RegularTypeCaptioner(),
                comparison_captioner=RegularTypeCaptioner()
            )
        )
        quantifier_captioner = QuantifierCaptioner(
            restrictor_captioner=RegularTypeCaptioner(
                hypernym_rate=1.0,
                logical_tautology_rate=1.0
            ),
            body_captioner=body_captioner
        )
        number_bound_captioner = NumberBoundCaptioner(
            quantifier_captioner=quantifier_captioner
        )
        comparative_quantifier_captioner = ComparativeQuantifierCaptioner(
            restrictor_captioner=RegularTypeCaptioner(
                hypernym_rate=1.0
            ),
            comparison_captioner=RegularTypeCaptioner(
                hypernym_rate=1.0
            ),
            body_captioner=body_captioner
        )
        quantification_captioner = CaptionerMixer(
            captioners=(quantifier_captioner, number_bound_captioner, comparative_quantifier_captioner),
            distribution=[2, 2, 1]
        )
        world_captioner = CaptionerMixer(
            captioners=(
                ConjunctionCaptioner(captioners=(existential_captioner, relation_captioner, quantification_captioner)),
                DisjunctionCaptioner(captioners=(existential_captioner, relation_captioner, quantification_captioner))
            )
        )

        super(Combination, self).__init__(
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


dataset = Combination
