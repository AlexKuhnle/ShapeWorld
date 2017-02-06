import os
from random import random, randrange
import numpy as np

from shapeworld.world import World


class Dataset(object):

    value_types = None
    default_config = None

    def __init__(self, world_generator):
        assert self.__class__.value_types is not None
        self.world_generator = world_generator

    @staticmethod
    def from_config(config=None, dataset_type=None, dataset_name=None, dataset_class=None):
        if config is not None:
            if isinstance(config, str):
                assert os.path.isfile(config)
                with open(config, 'r') as filehandle:
                    import json
                    config = json.load(fp=filehandle)
            if 'type' in config:
                dataset_type = config['type']
            if 'name' in config:
                dataset_name = config['name']
        if dataset_type and dataset_name:
            from importlib import import_module
            module = import_module('shapeworld.datasets.{}.{}'.format(dataset_type, dataset_name))
            dataset_class = module.dataset
        if config is None:
            assert dataset_class.default_config is not None
            config = dataset_class.default_config
        dataset = dataset_class(**config)
        return dataset

    @property
    def world_size(self):
        return self.world_generator.world_size

    @property
    def world_shape(self):
        return (self.world_generator.world_size[0], self.world_generator.world_size[1], 3)

    def generate(self, n, mode=None):  # mode: None, 'train', 'validation', 'test'
        raise NotImplementedError

    def iterate(self, n, mode=None):
        while True:
            yield self.generate(n=n, mode=mode)

    def serialize_data(self, directory, generated, predicted=None, additional=None):
        assert not additional or all(name not in self.__class__.value_types for name in additional)
        if os.path.isdir(directory):
            for root, dirs, files in os.walk(directory):
                assert root == directory
                assert not dirs
                for file in files:
                    path = os.path.join(root, file)
                    os.remove(path=path)
        else:
            os.makedirs(directory)

        id2word = {word_id: word for word, word_id in self.word_ids.items()} if hasattr(self, 'word_ids') else None
        for name in generated:
            Dataset.serialize_value(name=name, value=generated[name], value_type=self.__class__.value_types[name], directory=directory, id2word=id2word)
        if predicted:
            for name in predicted:
                Dataset.serialize_value(name='predicted_' + name, value=predicted[name], value_type=self.__class__.value_types[name], directory=directory, id2word=id2word)
        if additional:
            for name, (value, value_type) in additional.items():
                Dataset.serialize_value(name=name, value=value, value_type=value_type, directory=directory, id2word=id2word)

    @staticmethod
    def serialize_value(name, value, value_type, directory, id2word=None):
        if value_type == 'int':
            with open(os.path.join(directory, name + '.txt'), 'w') as filehandle:
                filehandle.write('\n'.join(str(int(x)) for x in value) + '\n')
        if value_type == 'index':
            with open(os.path.join(directory, name + '.txt'), 'w') as filehandle:
                for indices in value:
                    filehandle.write(' '.join(str(n) for n, x in enumerate(indices) if x) + '\n')
        elif value_type == 'image':
            for n in range(len(value)):
                image = World.get_image(world=value[n])
                image.save(fp=os.path.join(directory, '{}{}.bmp'.format(name, n)), format='bmp')
        elif value_type == 'text':
            with open(os.path.join(directory, name + '.txt'), 'w') as filehandle:
                for word_ids in value:
                    text = ' '.join(id2word[word_id] for word_id in word_ids if word_id > 0)
                    filehandle.write(text + '\n')


class ClassificationDataset(Dataset):

    value_types = {'world': 'image', 'class': 'int'}
    multi_class_flag = False

    def __init__(self, world_generator, class_count):
        super().__init__(world_generator)
        self.class_count = class_count

    def get_class(self, world):
        raise NotImplementedError

    def generate(self, n, mode=None):
        worlds = np.zeros(shape=(n, self.world_size.x, self.world_size.y, 3), dtype=np.float32)
        if self.__class__.multi_class_flag:
            classes = np.zeros(shape=(n, self.class_count), dtype=np.float32)
            for i in range(n):
                world = self.world_generator(mode)
                worlds[i] = world.get_world()
                for c in self.get_class(world):
                    classes[i][c] = 1.0
        else:
            classes = np.zeros(shape=(n, 1), dtype=np.int32)
            for i in range(n):
                world = self.world_generator(mode)
                worlds[i] = world.get_world()
                classes[i][0] = self.get_class(world)
        return {'world': worlds, 'class': classes}


class CaptionAgreementDataset(Dataset):

    value_types = {'world': 'image', 'caption': 'text', 'agreement': 'int'}

    def __init__(self, world_generator, world_captioner, incorrect_world_ratio=0.5, correct_ratio=0.5, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None):
        super().__init__(world_generator)
        self.world_captioner = world_captioner
        self.incorrect_world_ratio = incorrect_world_ratio
        self.correct_ratio = correct_ratio
        self.train_correct_ratio = correct_ratio if train_correct_ratio is None else train_correct_ratio
        self.validation_correct_ratio = correct_ratio if validation_correct_ratio is None else validation_correct_ratio
        self.test_correct_ratio = correct_ratio if test_correct_ratio is None else test_correct_ratio

    @property
    def caption_size(self):
        return self.world_captioner.caption_size

    @property
    def caption_shape(self):
        return (self.world_captioner.caption_size,)

    @property
    def word_ids(self):
        return self.world_captioner.word_ids

    @property
    def vocabulary_size(self):
        return len(self.world_captioner.word_ids)

    @property
    def vocabulary(self):
        return list(self.world_captioner.word_ids.keys())

    def generate_incorrect_world(self, world, caption, mode):
        if mode != 'train':
            mode = None
        while True:
            world = self.world_generator(mode)
            if caption.agreement(world) == 0.0:
                return world

    def generate_incorrect_caption(self, world, caption, mode):
        if mode != 'train':
            mode = None
        while True:
            incorrect_world = self.world_generator(mode)
            caption = self.world_captioner(incorrect_world, mode)
            if caption.agreement(world) == 0.0:
                return caption

    def generate(self, n, mode=None):
        if mode == 'train':
            correct_ratio = self.train_correct_ratio
        elif mode == 'validation':
            correct_ratio = self.validation_correct_ratio
        elif mode == 'test':
            correct_ratio = self.test_correct_ratio
        else:
            correct_ratio = self.correct_ratio
        worlds = np.zeros(shape=(n, self.world_size.x, self.world_size.y, 3), dtype=np.float32)
        captions = np.zeros(shape=(n, self.caption_size), dtype=np.int32)
        agreements = np.zeros(shape=(n, 1), dtype=np.float32)
        caption_list = []
        for i in range(n):
            world = self.world_generator(mode)
            caption = self.world_captioner(world, mode)
            if random() < correct_ratio:
                agreements[i][0] = 1.0
            elif random() < self.incorrect_world_ratio:
                world = self.generate_incorrect_world(world, caption, mode)
            else:
                caption = self.generate_incorrect_caption(world, caption, mode)
            worlds[i] = world.get_world()
            caption_list.append(caption)
        caption_list = self.world_captioner.realize(caption_list)
        for i, caption in enumerate(caption_list):
            assert len(caption) <= self.caption_size
            for j, word in enumerate(caption, start=(self.caption_size - len(caption))):
                captions[i][j] = self.word_ids[word]
        return {'world': worlds, 'caption': captions, 'agreement': agreements}


class MixerCaptionAgreementDataset(CaptionAgreementDataset):

    def __init__(self, datasets, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        super().__init__(datasets[0].world_generator, datasets[0].world_captioner)
        assert len(datasets) >= 1 and all(isinstance(dataset, CaptionAgreementDataset) for dataset in datasets)
        assert all(dataset.world_size == datasets[0].world_size for dataset in datasets)
        assert distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in distribution)
        assert train_distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in train_distribution)
        assert validation_distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in validation_distribution)
        assert test_distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in test_distribution)
        self.datasets = datasets
        self.distribution = distribution or [1.0 / len(datasets) for _ in range(len(datasets))]
        if sum(self.distribution) != 1.0:
            s = sum(self.distribution)
            self.distribution = [prob / s for prob in self.distribution]
        self.train_distribution = train_distribution or self.distribution
        if sum(self.train_distribution) != 1.0:
            s = sum(self.train_distribution)
            self.train_distribution = [prob / s for prob in self.train_distribution]
        self.validation_distribution = validation_distribution or self.distribution
        if sum(self.validation_distribution) != 1.0:
            s = sum(self.validation_distribution)
            self.validation_distribution = [prob / s for prob in self.validation_distribution]
        self.test_distribution = test_distribution or self.distribution
        if sum(self.test_distribution) != 1.0:
            s = sum(self.test_distribution)
            self.test_distribution = [prob / s for prob in self.test_distribution]
        caption_size = max(dataset.world_captioner.caption_size for dataset in datasets)
        words = set(word for dataset in datasets for word in dataset.word_ids)
        words = sorted(words)
        word_ids = {words[n]: n for n in range(len(words))}
        for dataset in self.datasets:
            dataset.world_captioner.caption_size = caption_size
            dataset.world_captioner.word_ids = word_ids

    def generate(self, n, mode=None):
        if mode == 'train':
            distribution = self.train_distribution
        elif mode == 'validation':
            distribution = self.validation_distribution
        elif mode == 'test':
            distribution = self.test_distribution
        else:
            distribution = self.distribution
        worlds = np.zeros(shape=(n, self.world_size.x, self.world_size.y, 3), dtype=np.float32)
        captions = np.zeros(shape=(n, self.caption_size), dtype=np.int32)
        agreements = np.zeros(shape=(n, 1), dtype=np.float32)
        for i in range(n):
            pick = random()
            cumulative = 0.0
            for dataset, prob in zip(self.datasets, distribution):
                cumulative += prob
                if pick < cumulative:
                    generated = dataset.generate(n=1, mode=mode)
                    worlds[i] = generated['world'][0]
                    captions[i] = generated['caption'][0]
                    agreements[i] = generated['agreement'][0]
                    break
            else:
                generated = self.datasets[-1].generate(n=1, mode=mode)
                worlds[i] = generated['world'][0]
                captions[i] = generated['caption'][0]
                agreements[i] = generated['agreement'][0]
        return {'world': worlds, 'caption': captions, 'agreement': agreements}


class ComparisonDataset(Dataset):

    value_types = {'reference': 'image', 'comparison': 'image', 'agreement': 'int'}
    correct_comparison_flag = None
    incorrect_comparison_flag = None

    def __init__(self, world_generator, correct_ratio, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None):
        super().__init__(world_generator)
        assert self.__class__.correct_comparison_flag is not None and self.__class__.incorrect_comparison_flag is not None
        self.correct_ratio = correct_ratio
        self.train_correct_ratio = correct_ratio if train_correct_ratio is None else train_correct_ratio
        self.validation_correct_ratio = correct_ratio if validation_correct_ratio is None else validation_correct_ratio
        self.test_correct_ratio = correct_ratio if test_correct_ratio is None else test_correct_ratio

    def generate_correct_comparison(self, reference, mode=None):
        if mode != 'train':
            mode = None
        if self.__class__.correct_comparison_flag:
            while True:
                comparison = self.world_generator(mode)
                if self.correct_comparison(reference, comparison):
                    break
            return comparison
        raise NotImplementedError

    def correct_comparison(self, reference, comparison):
        raise NotImplementedError

    def generate_incorrect_comparison(self, reference, mode=None):
        if mode != 'train':
            mode = None
        if self.__class__.incorrect_comparison_flag:
            while True:
                comparison = self.world_generator(mode)
                if self.incorrect_comparison(reference, comparison):
                    break
            return comparison
        raise NotImplementedError

    def incorrect_comparison(self, reference, comparison):
        raise NotImplementedError

    def generate(self, n, mode=None):
        if mode == 'train':
            correct_ratio = self.train_correct_ratio
        elif mode == 'validation':
            correct_ratio = self.validation_correct_ratio
        elif mode == 'test':
            correct_ratio = self.test_correct_ratio
        else:
            correct_ratio = self.correct_ratio
        references = np.zeros(shape=(n, self.world_size.x, self.world_size.y, 3), dtype=np.float32)
        comparisons = np.zeros(shape=(n, self.world_size.x, self.world_size.y, 3), dtype=np.float32)
        agreements = np.zeros(shape=(n, 1), dtype=np.float32)
        for i in range(n):
            reference = self.world_generator(mode)
            references[i] = reference.get_world()
            if random() < correct_ratio:
                comparison = self.generate_correct_comparison(reference)
                agreements[i][0] = 1.0
            else:
                comparison = self.generate_incorrect_comparison(reference)
            comparisons[i] = comparison.get_world()
        return {'reference': references, 'comparison': comparisons, 'agreement': agreements}


class CommunicationDataset(Dataset):

    value_types = {'alternative1': 'image', 'alternative2': 'image', 'reference': 'int'}
    valid_alternative_flag = None

    def __init__(self, world_generator, alternative_count=2):
        super().__init__(world_generator)
        assert self.__class__.valid_alternative_flag is not None
        self.alternative_count = alternative_count

    def generate_alternative(self, reference, mode=None):
        if mode != 'train':
            mode = None
        if self.__class__.valid_alternative_flag:
            while True:
                alternative = self.world_generator(mode)
                if self.valid_alternative(reference, alternative):
                    break
            return alternative
        raise NotImplementedError

    def valid_alternative(self, reference, alternative):
        raise NotImplementedError

    def generate(self, n, mode=None):
        alternatives = [np.zeros(shape=(n, self.world_size.x, self.world_size.y, 3), dtype=np.float32) for _ in range(self.alternative_count)]
        references = np.zeros(shape=(n, 1), dtype=np.int32)
        for i in range(n):
            reference = randrange(self.alternative_count)
            references[i] = reference
            world = self.world_generator(mode)
            alternatives[reference][i] = world.get_world()
            for alt in range(self.alternative_count):
                if alt == reference:
                    continue
                alternative = self.generate_alternative(world, mode)
                alternatives[alt][i] = alternative.get_world()
        result = {'alternative' + str(alt + 1): alternatives[alt] for alt in range(self.alternative_count)}
        result['reference'] = references
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate example data')
    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
    parser.add_argument('-c', '--config', default=None, help='Dataset configuration file')
    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test'), help='Mode')
    parser.add_argument('-i', '--instances', type=int, default=100, help='Number of instances')
    parser.add_argument('-d', '--directory', default='examples', help='Directory for generated data')
    args = parser.parse_args()

    dataset = Dataset.from_config(config=args.config, dataset_type=args.type, dataset_name=args.name)
    directory = os.path.join(args.directory, args.type, args.name)
    generated = dataset.generate(n=args.instances, mode=args.mode)
    dataset.serialize_data(directory=directory, generated=generated)
