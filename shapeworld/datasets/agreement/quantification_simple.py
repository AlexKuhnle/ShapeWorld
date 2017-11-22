from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator, LimitedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, QuantifierCaptioner, NumberBoundCaptioner, ComparativeQuantifierCaptioner


class QuantificationSimple(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=15,
        vocabulary=('.', 'a', 'above', 'all', 'almost', 'an', 'are', 'as', 'at', 'behind', 'below', 'bigger', 'biggest', 'black', 'blue', 'both', 'but', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'darkest', 'eight', 'ellipse', 'ellipses', 'exactly', 'farther', 'farthest', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'in', 'is', 'least', 'left', 'leftmost', 'less', 'lighter', 'lightest', 'lowermost', 'magenta', 'many', 'more', 'most', 'no', 'none', 'not', 'of', 'one', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red', 'right', 'rightmost', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'smallest', 'square', 'squares', 'than', 'the', 'third', 'thirds', 'three', 'to', 'topmost', 'triangle', 'triangles', 'twice', 'two', 'yellow', 'zero'),
        language=None
    ):

        # world_generator = LimitedAttributesGenerator(
        #     shapes_range=(2, 4),
        #     colors_range=(2, 4),
        #     textures_range=(1, 1),
        world_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 3),
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
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
        world_captioner = CaptionerMixer(
            captioners=(quantifier_captioner, number_bound_captioner, comparative_quantifier_captioner),
            distribution=[2, 2, 1]
        )

        super(QuantificationSimple, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = QuantificationSimple
