from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, RelationCaptioner, QuantifierCaptioner, NumberBoundCaptioner, ComparativeQuantifierCaptioner


class QuantificationCount(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=19,
        vocabulary=('.', 'a', 'above', 'all', 'an', 'are', 'as', 'at', 'behind', 'below', 'bigger', 'biggest', 'blue', 'both', 'but', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'darkest', 'eight', 'ellipse', 'ellipses', 'exactly', 'farther', 'farthest', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'in', 'is', 'least', 'left', 'leftmost', 'less', 'lighter', 'lightest', 'lowermost', 'magenta', 'many', 'more', 'most', 'not', 'of', 'one', 'pentagon', 'pentagons', 'rectangle', 'rectangles', 'red', 'right', 'rightmost', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'smallest', 'square', 'squares', 'than', 'the', 'three', 'to', 'topmost', 'triangle', 'triangles', 'twice', 'two', 'yellow', 'zero'),
        language=None
    ):

        world_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 3),
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )

        body_captioner = CaptionerMixer(
            captioners=(
                AttributeTypeRelationCaptioner(
                    attribute_type_captioner=CaptionerMixer(
                        captioners=(
                            RegularAttributeCaptioner(),
                            RegularTypeCaptioner()
                        )
                    )
                ),
                RelationCaptioner(
                    reference_captioner=RegularTypeCaptioner(),
                    comparison_captioner=RegularTypeCaptioner()
                )
            ),
            distribution=[1, 2]
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

        super(QuantificationCount, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = QuantificationCount
