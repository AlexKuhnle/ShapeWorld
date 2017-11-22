from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator, LimitedAttributesGenerator  # , ClusteredEntitiesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, QuantifierCaptioner, NumberBoundCaptioner, ComparativeQuantifierCaptioner


class QuantificationCountSimple(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=13,
        vocabulary=('.', 'a', 'all', 'an', 'are', 'as', 'at', 'blue', 'but', 'circle', 'circles', 'cross', 'crosses', 'cyan', 'eight', 'ellipse', 'ellipses', 'exactly', 'five', 'four', 'gray', 'green', 'half', 'is', 'least', 'less', 'magenta', 'many', 'more', 'most', 'not', 'of', 'one', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'square', 'squares', 'than', 'the', 'three', 'triangle', 'triangles', 'twice', 'two', 'yellow', 'zero'),
        language=None
    ):

        # world_generator = LimitedAttributesGenerator(
        #     shapes_range=(2, 4),
        #     colors_range=(2, 4),
        #     textures_range=(1, 1),
        world_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 3),
            entity_counts=entity_counts,
            collision_tolerance=0.0,
            boundary_tolerance=0.0,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations,
            max_provoke_collision_rate=0.0
        )

        body_captioner = AttributeTypeRelationCaptioner(
            attribute_type_captioner=CaptionerMixer(
                captioners=(
                    RegularAttributeCaptioner(),
                    RegularTypeCaptioner()
                )
            )
        )
        quantifier_captioner = QuantifierCaptioner(
            restrictor_captioner=RegularTypeCaptioner(
                hypernym_rate=1.0,
                logical_tautology_rate=1.0
            ),
            body_captioner=body_captioner,
            quantifiers=('count',)
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
        world_captioner = CaptionerMixer(
            captioners=(quantifier_captioner, number_bound_captioner, comparative_quantifier_captioner),
            distribution=[1, 1, 1]
        )

        super(QuantificationCountSimple, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = QuantificationCountSimple
