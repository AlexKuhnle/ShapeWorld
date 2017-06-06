from shapeworld.dataset import CaptionAgreementDataset
from shapeworld.generators import GenericGenerator
from shapeworld.captioners import ExistentialCaptioner


class MultiShapeDataset(CaptionAgreementDataset):

    def __init__(self, entity_counts, train_entity_counts, validation_entity_counts, test_entity_counts, caption_size, words, incorrect_caption_distribution=None, hypernym_ratio=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, realizer=None, world_size=None, world_color=None, shapes=None, colors=None, textures=None, rotation=None, size_range=None, distortion_range=None, shade_range=None, noise_range=None, collision_tolerance=None, boundary_tolerance=None, quantifier_tolerance=None, **kwargs):
        world_generator = GenericGenerator(entity_counts, world_size, world_color, shapes, colors, textures, rotation, size_range, distortion_range, shade_range, noise_range, collision_tolerance, boundary_tolerance, train_entity_counts=train_entity_counts, validation_entity_counts=validation_entity_counts, test_entity_counts=test_entity_counts)
        world_captioner = ExistentialCaptioner(world_generator.shapes, world_generator.colors, world_generator.textures, quantifier_tolerance=quantifier_tolerance, incorrect_distribution=incorrect_caption_distribution, hypernym_ratio=hypernym_ratio)
        super(MultiShapeDataset, self).__init__(
            world_generator=world_generator,
            world_captioner=world_captioner,
            caption_size=caption_size,
            words=words,
            incorrect_world_ratio=0.0,
            correct_ratio=correct_ratio,
            train_correct_ratio=correct_ratio,
            validation_correct_ratio=validation_correct_ratio,
            test_correct_ratio=test_correct_ratio,
            caption_realizer=realizer)


dataset = MultiShapeDataset
MultiShapeDataset.dataset_name = 'multishape'
MultiShapeDataset.default_config = {
    'entity_counts': [1, 2, 3, 4, 5, 6],
    'train_entity_counts': [1, 2, 3, 5],
    'validation_entity_counts': [4],
    'test_entity_counts': [6],
    'caption_size': 6,
    'words': ['.', 'a', 'an', 'black', 'blue', 'circle', 'cross', 'cyan', 'ellipse', 'green', 'is', 'magenta', 'pentagon', 'rectangle', 'red', 'semicircle', 'shape', 'square', 'there', 'triangle', 'white', 'yellow']
}
