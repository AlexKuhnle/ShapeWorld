from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import ReinforcedAttributesGenerator
from shapeworld.captioners import RegularTypeCaptioner, RelationCaptioner, ExistentialCaptioner


class Relational(CaptionAgreementDataset):

    def __init__(
        self,
        entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        train_entity_counts=(5, 6, 7, 8, 9, 10, 11, 12, 14),
        validation_entity_counts=(13,),
        test_entity_counts=(15,),
        validation_combinations=(('square', 'red', 'solid'), ('triangle', 'green', 'solid'), ('circle', 'blue', 'solid')),
        test_combinations=(('rectangle', 'yellow', 'solid'), ('cross', 'magenta', 'solid'), ('ellipse', 'cyan', 'solid')),
        caption_size=14,
        vocabulary=('.', 'a', 'above', 'an', 'behind', 'below', 'bigger', 'biggest', 'blue', 'circle', 'closer', 'closest', 'cross', 'cyan', 'darker', 'darkest', 'ellipse', 'farther', 'farthest', 'from', 'front', 'gray', 'green', 'in', 'is', 'left', 'leftmost', 'lighter', 'lightest', 'lowermost', 'magenta', 'most', 'of', 'pentagon', 'rectangle', 'red', 'right', 'rightmost', 'semicircle', 'shape', 'smaller', 'smallest', 'square', 'than', 'the', 'to', 'topmost', 'triangle', 'yellow'),
        language=None
    ):

        world_generator = ReinforcedAttributesGenerator(
            reinforcement_range=(1, 1),
            entity_counts=entity_counts,
            train_entity_counts=train_entity_counts,
            validation_entity_counts=validation_entity_counts,
            test_entity_counts=test_entity_counts,
            validation_combinations=validation_combinations,
            test_combinations=test_combinations
        )

        world_captioner = ExistentialCaptioner(
            restrictor_captioner=RegularTypeCaptioner(),
            body_captioner=RelationCaptioner(
                reference_captioner=RegularTypeCaptioner(),
                comparison_captioner=RegularTypeCaptioner()
            )
        )

        super(Relational, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            vocabulary=vocabulary,
            language=language
        )


dataset = Relational
