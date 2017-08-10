from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import CaptionerMixer, AttributesNounCaptioner, AttributesRelationCaptioner, ExistentialCaptioner


class OneShapeDataset(CaptionAgreementDataset):

    dataset_name = 'oneshape'

    def __init__(self, validation_combinations, test_combinations, caption_size, words, hypernym_ratio=None, incorrect_caption_distribution=None, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, realizer=None, language=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, collision_tolerance=None, boundary_tolerance=None, **kwargs):
        world_generator = GenericGenerator([1], world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, collision_tolerance, boundary_tolerance, validation_combinations=validation_combinations, test_combinations=test_combinations)
        world_captioner = CaptionerMixer(
            captioners=(
                AttributesNounCaptioner(
                    incorrect_distribution=incorrect_caption_distribution
                ),
                ExistentialCaptioner(
                    subject_captioner=AttributesNounCaptioner(hypernym_ratio=1.0),
                    verb_captioner=AttributesRelationCaptioner(),
                    incorrect_distribution=incorrect_caption_distribution
                )
            ),
            distribution=distribution,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            test_distribution=test_distribution
        )
        super(OneShapeDataset, self).__init__(
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


dataset = OneShapeDataset
OneShapeDataset.default_config = {
    'validation_combinations': [['square', 'red', 'solid'], ['triangle', 'green', 'solid'], ['circle', 'blue', 'solid']],
    'test_combinations': [['rectangle', 'yellow', 'solid'], ['cross', 'magenta', 'solid'], ['ellipse', 'cyan', 'solid']],
    'caption_size': 8,
    'words': ['.', 'a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'white', 'yellow']
}
