from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GeneratorMixer, RandomAttributesGenerator, ReinforcedAttributesGenerator
from shapeworld.captioners import CaptionerMixer, RegularAttributeCaptioner, RegularTypeCaptioner, AttributeTypeRelationCaptioner, RelationCaptioner, ExistentialCaptioner, QuantifierCaptioner, NumberBoundCaptioner, ComparativeQuantifierCaptioner, ConjunctionCaptioner, DisjunctionCaptioner


class Combination(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=28,
        vocabulary=('.', 'a', 'above', 'all', 'an', 'and', 'are', 'at', 'behind', 'below', 'bigger', 'biggest', 'black', 'blue', 'both', 'circle', 'circles', 'closer', 'closest', 'cross', 'crosses', 'cyan', 'darker', 'darkest', 'eight', 'either', 'ellipse', 'ellipses', 'every', 'exactly', 'farther', 'farthest', 'few', 'five', 'four', 'from', 'front', 'gray', 'green', 'half', 'in', 'is', 'least', 'left', 'leftmost', 'less', 'lighter', 'lightest', 'lowermost', 'magenta', 'more', 'most', 'no', 'not', 'of', 'one', 'or', 'pentagon', 'pentagons', 'quarter', 'quarters', 'rectangle', 'rectangles', 'red', 'right', 'rightmost', 'semicircle', 'semicircles', 'seven', 'shape', 'shapes', 'six', 'smaller', 'smallest', 'some', 'square', 'squares', 'than', 'the', 'there', 'third', 'three', 'to', 'topmost', 'triangle', 'triangles', 'two', 'yellow', 'zero'),
        language=None
    ):

        random_generator = RandomAttributesGenerator(
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )
        reinforced_attributes_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 3),
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
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
        multishape_captioner = CaptionerMixer(
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
        relational_captioner = ExistentialCaptioner(
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
                ConjunctionCaptioner(captioners=(multishape_captioner, relational_captioner, quantification_captioner)),
                DisjunctionCaptioner(captioners=(multishape_captioner, relational_captioner, quantification_captioner))
            )
        )

        super(Combination, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = Combination
