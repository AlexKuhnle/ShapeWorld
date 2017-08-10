from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import AttributesNounCaptioner, SpatialRelationCaptioner, ExistentialCaptioner


class SpatialSimpleDataset(CaptionAgreementDataset):

    dataset_name = 'spatial_simple'

    def __init__(self, validation_combinations, test_combinations, caption_size, words, incorrect_caption_distribution=None, hypernym_ratio=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, realizer=None, language=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, quantifier_tolerance=None, **kwargs):
        world_generator = GenericGenerator([2], world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance, validation_combinations=validation_combinations, test_combinations=test_combinations)
        world_captioner = ExistentialCaptioner(
            subject_captioner=AttributesNounCaptioner(),
            verb_captioner=SpatialRelationCaptioner(relations=('x-rel', 'y-rel')),
            incorrect_distribution=incorrect_caption_distribution
        )
        super(SpatialSimpleDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            incorrect_world_ratio=0.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio,
            caption_realizer=realizer,
            language=language
        )


dataset = SpatialSimpleDataset
SpatialSimpleDataset.default_config = {
    'validation_combinations': [['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    'test_combinations': [['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    'caption_size': 12,
    'words': ['.', 'a', 'above', 'an', 'below', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'left', 'magenta', 'of', 'pentagon', 'rectangle', 'red', 'right', 'semicircle', 'shape', 'square', 'the', 'to', 'triangle', 'white', 'yellow']
}
