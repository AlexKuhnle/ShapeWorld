from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, RelationCaptioner, QuantifierCaptioner, NumberBoundCaptioner


class QuantificationRatio(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=18,
        vocabulary=('.', 'a', 'above', 'all', 'an', 'are', 'behind', 'below', 'bigger', 'biggest', 'blue', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'darkest', 'eight', 'ellipse', 'ellipses', 'farther', 'farthest', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'in', 'is', 'left', 'leftmost', 'lighter', 'lightest', 'lowermost', 'magenta', 'most', 'no', 'none', 'of', 'one', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red', 'right', 'rightmost', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'smallest', 'square', 'squares', 'than', 'the', 'third', 'thirds', 'three', 'to', 'topmost', 'triangle', 'triangles', 'two', 'yellow'),
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
            quantifiers=('ratio',)
        )
        number_bound_captioner = NumberBoundCaptioner(
            quantifier_captioner=quantifier_captioner
        )
        world_captioner = CaptionerMixer(
            captioners=(quantifier_captioner, number_bound_captioner),
            distribution=[1, 1]
        )

        super(QuantificationRatio, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language)


dataset = QuantificationRatio
