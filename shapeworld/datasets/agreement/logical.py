from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GeneratorMixer, RandomAttributesGenerator, ReinforcedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, EmptyTypeCaptioner, RegularAttributeCaptioner, RegularTypeCaptioner, UniqueTypeCaptioner, SelectorCaptioner, AttributeTypeRelationCaptioner, RelationCaptioner, ExistentialCaptioner, QuantifierCaptioner, ConjunctionCaptioner, DisjunctionCaptioner, ImplicationCaptioner, EquivalenceCaptioner


class LogicalDataset(CaptionAgreementDataset):

    def __init__(
        self,
        world_size=64,
        world_colors=('black',),
        shapes=('square', 'rectangle', 'triangle', 'pentagon', 'cross', 'circle', 'semicircle', 'ellipse'),
        colors=('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'gray'),
        textures=('solid',),
        rotation=True,
        size_range=(0.1, 0.25),
        distortion_range=(2.0, 3.0),
        shade_range=0.4,
        collision_tolerance=0.25,
        collision_shade_difference=0.5,
        boundary_tolerance=None,
        entity_counts=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=None,
        validation_entity_counts=None,
        test_entity_counts=None,
        validation_count_rate=0.5,
        test_count_rate=0.5,
        validation_combinations=None,
        test_combinations=None,
        validation_space_rate_range=(0.0, 1.0),
        test_space_rate_range=(0.0, 1.0),
        validation_combination_rate=0.5,
        test_combination_rate=0.5,
        max_provoke_collision_rate=0.33,
        reinforcement_range=(1, 3),
        generators=None,
        captioners=None,
        connectives=None,
        caption_size=31,
        vocabulary=('.', 'a', 'above', 'all', 'an', 'and', 'are', 'as', 'at', 'behind', 'below', 'besides', 'bigger', 'biggest', 'blue', 'but', 'circle', 'circles', 'closer', 'closest', 'color', 'cross', 'crosses', 'cyan', 'darker', 'darkest', 'different', 'eight', 'either', 'ellipse', 'ellipses', 'exactly', 'exists', 'farther', 'farthest', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'if', 'in', 'is', 'least', 'left', 'leftmost', 'less', 'lighter', 'lightest', 'lower', 'lowermost', 'magenta', 'many', 'more', 'most', 'no', 'none', 'not', 'of', 'one', 'only', 'or', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red', 'right', 'rightmost', 'same', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'smallest', 'square', 'squares', 'than', 'the', 'there', 'third', 'thirds', 'three', 'to', 'triangle', 'triangles', 'twice', 'two', 'upper', 'uppermost', 'yellow', 'zero'),
        correct_ratio=0.5,
        train_correct_ratio=None,
        validation_correct_ratio=None,
        test_correct_ratio=None,
        worlds_per_instance=1,
        captions_per_instance=1,
        pixel_noise_stddev=None,
        caption_realizer='dmrs',
        language=None
    ):

        generator_list = list()

        if generators is None or 'random' in generators:
            random_generator = RandomAttributesGenerator(
                world_size=world_size,
                world_colors=world_colors,
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
                validation_count_rate=validation_count_rate,
                test_entity_counts=test_entity_counts,
                test_count_rate=test_count_rate,
                validation_combinations=validation_combinations,
                validation_space_rate_range=validation_space_rate_range,
                validation_combination_rate=validation_combination_rate,
                test_combinations=test_combinations,
                test_space_rate_range=test_space_rate_range,
                test_combination_rate=test_combination_rate,
                max_provoke_collision_rate=max_provoke_collision_rate
            )
            generator_list.append(random_generator)

        if generators is None or 'reinforced' in generators:
            reinforced_generator = ReinforcedAttributesGenerator(
                world_size=world_size,
                world_colors=world_colors,
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
                validation_count_rate=validation_count_rate,
                test_entity_counts=test_entity_counts,
                test_count_rate=test_count_rate,
                validation_combinations=validation_combinations,
                validation_space_rate_range=validation_space_rate_range,
                validation_combination_rate=validation_combination_rate,
                test_combinations=test_combinations,
                test_space_rate_range=test_space_rate_range,
                test_combination_rate=test_combination_rate,
                max_provoke_collision_rate=max_provoke_collision_rate,
                reinforcement_range=reinforcement_range
            )
            generator_list.append(reinforced_generator)

        world_generator = GeneratorMixer(
            generators=generator_list
        )

        restrictor_captioner = CaptionerMixer(
            captioners=(
                EmptyTypeCaptioner(),
                RegularTypeCaptioner(hypernym_rate=1.0)
            )
        )

        body_captioner = AttributeTypeRelationCaptioner(
            attribute_type_captioner=CaptionerMixer(
                captioners=(
                    RegularAttributeCaptioner(),
                    RegularTypeCaptioner(hypernym_rate=0.0)
                )
            )
        )

        captioner_list = list()

        if captioners is None or 'existential' in captioners:
            existential_captioner = CaptionerMixer(
                captioners=(
                    RegularTypeCaptioner(existing_attribute_rate=0.0),
                    ExistentialCaptioner(
                        restrictor_captioner=restrictor_captioner,
                        body_captioner=AttributeTypeRelationCaptioner(
                            attribute_type_captioner=CaptionerMixer(
                                captioners=(
                                    RegularAttributeCaptioner(existing_attribute_rate=0.0),
                                    RegularTypeCaptioner(hypernym_rate=0.0, existing_attribute_rate=0.0)
                                )
                            )
                        )
                    )
                )
            )
            captioner_list.append(existential_captioner)

        if captioners is None or 'relational' in captioners:
            relational_captioner = ExistentialCaptioner(
                restrictor_captioner=RegularTypeCaptioner(),
                body_captioner=RelationCaptioner(
                    reference_captioner=RegularTypeCaptioner(),
                    comparison_captioner=UniqueTypeCaptioner()
                )
            )
            captioner_list.append(relational_captioner)

        if captioners is None or 'selection' in captioners:
            selection_captioner = ExistentialCaptioner(
                restrictor_captioner=SelectorCaptioner(
                    scope_captioner=restrictor_captioner,
                    comparison_captioner=UniqueTypeCaptioner()
                ),
                body_captioner=body_captioner
            )
            captioner_list.append(selection_captioner)

        if captioners is None or 'quantification' in captioners:
            quantification_captioner = QuantifierCaptioner(
                restrictor_captioner=restrictor_captioner,
                body_captioner=body_captioner
            )
            captioner_list.append(quantification_captioner)

        captioner = CaptionerMixer(
            captioners=captioner_list
        )

        captioner_list = list()

        if connectives is None or 'conjunction' in connectives:
            captioner_list.append(ConjunctionCaptioner(captioner=captioner))

        if connectives is None or 'disjunction' in connectives:
            captioner_list.append(DisjunctionCaptioner(captioner=captioner))

        if connectives is None or 'implication' in connectives:
            captioner_list.append(ImplicationCaptioner(captioner=captioner))

        if connectives is None or 'equivalence' in connectives:
            captioner_list.append(EquivalenceCaptioner(captioner=captioner))

        world_captioner = CaptionerMixer(
            captioners=captioner_list
        )

        super(LogicalDataset, self).__init__(
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
