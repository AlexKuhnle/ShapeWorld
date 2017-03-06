from io import BytesIO
import json
import os
from random import random, randrange
import numpy as np
from PIL import Image
from shapeworld.world import World
from shapeworld.archive import Archive


class Dataset(object):

    name = None
    value_types = None
    default_config = None

    def __init__(self, world_generator):
        assert self.__class__.name
        assert self.__class__.value_types and all(value_type in ('int', 'float', 'vector', 'text', 'world', 'model') for value_type in self.__class__.value_types.values())
        assert self.__class__.default_config
        self.world_generator = world_generator

    @staticmethod
    def from_config(config=None, dataset_type=None, dataset_name=None, dataset_class=None):  # if type = 'load'...
        load = (dataset_type == 'load') or (dataset_name == 'load')
        if config is not None:
            if isinstance(config, str):
                if load and os.path.isdir(config):
                    config = os.path.join(config, 'specification.json')
                assert os.path.isfile(config)
                with open(config, 'r') as filehandle:
                    config = json.load(fp=filehandle)
        if load:
            # load dataset with directory, then automatically find specification file
            return LoadDataset(specification=config)
        if config is not None:
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

    def __str__(self):
        return self.__class__.name

    @property
    def value_types(self):
        return self.__class__.value_types

    @property
    def world_size(self):
        return self.world_generator.world_size

    @property
    def text_size(self):
        return None

    @property
    def word_ids(self):
        return None

    def specification(self):
        return {'name': str(self),
                'value_types': self.value_types,
                'world_size': self.world_size,
                'text_size': self.text_size,
                'word_ids': self.word_ids}

    @property
    def world_shape(self):
        return (self.world_size, self.world_size, 3)

    @property
    def text_shape(self):
        return (self.text_size,) if self.text_size else None

    @property
    def vocabulary_size(self):
        return len(self.word_ids)

    @property
    def vocabulary(self):
        return list(self.word_ids.keys())

    def generate(self, n, mode=None, noise=True, include_model=False):  # mode: None, 'train', 'validation', 'test'
        raise NotImplementedError

    def iterate(self, n, mode=None, noise=True, include_model=False):
        while True:
            yield self.generate(n=n, mode=mode, noise=noise, include_model=include_model)

    def serialize_data(self, directory, generated, predicted=None, additional=None, archive=None, tiff=False):
        assert not additional or all(name not in self.value_types for name in additional)
        if os.path.isdir(directory):
            for root, dirs, files in os.walk(directory):
                assert root == directory
                assert not dirs
                assert not files
        else:
            os.makedirs(directory)

        id2word = [word for word, _ in sorted(self.word_ids.items(), key=(lambda kv: kv[1]))] if self.word_ids else None
        temp_path = os.path.join(directory, 'temp')

        with Archive(directory=directory, mode='w', archive=archive) as write_file:
            for name in generated:
                Dataset.serialize_value(value=generated[name], value_name=name, value_type=self.value_types[name], write_file=write_file, id2word=id2word, tiff=tiff, temp_path=temp_path)
            if predicted:
                for name in predicted:
                    Dataset.serialize_value(value=predicted[name], value_name='predicted_' + name, value_type=self.value_types[name], write_file=write_file, id2word=id2word, tiff=tiff, temp_path=temp_path)
            if additional:
                for name, (value, value_type) in additional.items():
                    Dataset.serialize_value(value=value, value_name=name, value_type=value_type, write_file=write_file, id2word=id2word, tiff=tiff, temp_path=temp_path)

    @staticmethod
    def serialize_value(value, value_name, value_type, write_file, id2word=None, tiff=False, temp_path=None):
        if value_type == 'int':
            value = '\n'.join(str(int(x)) for x in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'float':
            value = '\n'.join(str(float(x)) for x in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'vector':
            value = '\n'.join(','.join(str(x) for x in vector) for vector in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'text':
            assert id2word
            value = '\n'.join(' '.join(id2word[word_id] for word_id in text) for text in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'world':
            if tiff:
                assert temp_path
                from PIL import TiffImagePlugin
                TiffImagePlugin.WRITE_LIBTIFF = True
            for n in range(len(value)):
                image = World.get_image(world=value[n])
                if tiff:
                    image.save(temp_path, format='tiff', compression='tiff_lzw')
                    with open(temp_path, 'rb') as filehandle:
                        image_bytes = filehandle.read()
                    write_file('{}-{}.tiff'.format(value_name, n), image_bytes, binary=True)
                else:
                    image_bytes = BytesIO()
                    image.save(image_bytes, format='bmp')
                    write_file('{}-{}.bmp'.format(value_name, n), image_bytes.getvalue(), binary=True)
                    image_bytes.close()
            if tiff:
                TiffImagePlugin.WRITE_LIBTIFF = False
        elif value_type == 'model':
            value = json.dumps(value)
            write_file(value_name + '.json', value)

    @staticmethod
    def deserialize_value(value_name, value_type, read_file, word2id=None, tiff=False):
        if value_type == 'int':
            value = read_file(value_name + '.txt')
            value = [int(x) for x in value.split()]
            return value
        elif value_type == 'float':
            value = read_file(value_name + '.txt')
            value = [float(x) for x in value.split()]
            return value
        elif value_type == 'vector':
            value = read_file(value_name + '.txt')
            value = [[float(x) for x in vector.split(',')] for vector in value.split()]
            return value
        elif value_type == 'text':
            assert word2id
            value = read_file(value_name + '.txt')
            value = [[word2id[word] for word in text.split(' ')] for text in value.split('\n')[:-1]]
            return value
        elif value_type == 'world':
            if tiff:
                from PIL import TiffImagePlugin
                TiffImagePlugin.WRITE_LIBTIFF = True
            value = []
            n = 0
            while True:
                image_bytes = read_file('{}-{}.{}'.format(value_name, n, 'tiff' if tiff else 'bmp'), binary=True)
                if image_bytes is None:
                    break
                image_bytes = BytesIO(image_bytes)
                if tiff:
                    image = Image.open(image_bytes)  # , format='tiff', compression='tiff_lzw')
                else:
                    image = Image.open(image_bytes)  # , format='bmp')
                value.append(World.from_image(image))
                n += 1
            if tiff:
                TiffImagePlugin.WRITE_LIBTIFF = False
            return value
        elif value_type == 'model':
            value = read_file(value_name + '.json')
            value = json.loads(value)
            return value


class LoadDataset(Dataset):

    def __init__(self, specification, per_batch=True, batch_once=False):
        assert not batch_once or per_batch
        self._name = specification['name']
        self._value_types = specification['value_types']
        self._world_size = specification['world_size']
        self._text_size = specification.get('text_size')
        self._word_ids = specification.get('word_ids')
        self.world_model = specification.get('world_model')
        self.noise_range = specification.get('noise_range')
        self.tiff = specification.get('tiff')
        self.archive = specification.get('archive')
        self.per_batch = specification.get('per_batch', per_batch)
        self.batch_once = specification.get('batch_once', batch_once)

        self.batches = dict()
        directory = specification['directory']
        assert os.path.isdir(directory)
        for root, dirs, files in os.walk(directory):
            if root == directory:
                assert len(files) == 1 and 'specification.json' in files
                assert len(dirs) == 3 and 'train' in dirs and 'validation' in dirs and 'test' in dirs
            elif root[len(directory) + 1:] in ('train', 'validation', 'test'):
                mode = root[len(directory) + 1:]
                assert bool(dirs) != bool(files)
                if dirs:
                    assert all(d[:5] == 'batch' and d[5:].isdigit() for d in dirs)
                    self.batches[mode] = [os.path.join(root, d) for d in dirs]
                else:
                    self.batches[mode] = [root]
        assert self.batches
        self.mode = None
        self.values = {value_name: [] for value_name, value_type in self.value_types.items() if value_type != 'model' or self.world_model}
        self.num_instances = 0

    def __str__(self):
        return 'Loaded({})'.format(self._name)

    @property
    def value_types(self):
        return self._value_types

    @property
    def world_size(self):
        return self._world_size

    @property
    def text_size(self):
        return self._text_size

    @property
    def word_ids(self):
        return self._word_ids

    def zero_batch(self, n, include_model=False):
        batch = dict()
        for value_name, value_type in self.value_types.items():
            if value_type == 'int':
                batch[value_name] = np.zeros(shape=(n, 1), dtype=np.int32)
            elif value_type == 'float':
                batch[value_name] = np.zeros(shape=(n, 1), dtype=np.float32)
            elif value_type == 'vector':
                batch[value_name] = np.zeros(shape=(n, len(self.values[value_name][0])), dtype=np.float32)
            elif value_type == 'text':
                batch[value_name] = np.zeros(shape=(n, self.text_size), dtype=np.int32)
            elif value_type == 'world':
                batch[value_name] = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
            elif value_type == 'model' and include_model:
                batch[value_name] = [None] * n
        return batch

    def generate(self, n, mode=None, noise=True, include_model=False):
        assert noise or self.noise_range
        assert not include_model or self.world_model
        while not self.per_batch or self.mode != mode or self.num_instances < n:
            if self.mode != mode:
                self.mode = mode
                self.values = {value_name: [] for value_name in self.values}
            batches = self.batches[mode]
            batch = randrange(len(batches))
            batch_directory = batches.pop(batch) if self.batch_once else batches[batch]
            self.num_instances = 0
            with Archive(directory=batch_directory, mode='r', archive=self.archive) as read_file:
                for value_name, value in self.values.items():
                    value.extend(Dataset.deserialize_value(value_name=value_name, value_type=self.value_types[value_name], read_file=read_file, word2id=self.word_ids, tiff=self.tiff))
                    if self.num_instances:
                        assert len(value) == self.num_instances
                    else:
                        self.num_instances = len(value)
            temp_path = os.path.join(batch_directory, 'temp')
            if os.path.isfile(temp_path):
                os.remove(temp_path)
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            index = randrange(self.num_instances)
            self.num_instances -= 1
            for value_name, value_type in self.value_types.items():
                if value_type != 'model' or include_model:
                    batch[value_name][i] = self.values[value_name].pop(index)
        if noise and self.noise_range:
            for value_name, value_type in self.value_types.items():
                if value_type == 'world':
                    noise = np.random.normal(loc=0.0, scale=self.noise_range, size=(n, self.world_size, self.world_size, 3))
                    mask = (noise < -self.noise_range) + (noise > self.noise_range)
                    while np.any(a=mask):
                        noise -= mask * noise
                        noise += mask * np.random.normal(loc=0.0, scale=self.noise_range, size=(n, self.world_size, self.world_size, 3))
                        mask = (noise < -self.noise_range) + (noise > self.noise_range)
                    worlds = batch[value_name]
                    worlds += noise
                    np.clip(worlds, a_min=0.0, a_max=1.0, out=worlds)
        return batch


class ClassificationDataset(Dataset):

    value_types = {'world': 'world', 'world-model': 'model', 'classification': 'vector'}
    multi_class_flag = False
    class_count_flag = False

    def __init__(self, world_generator, num_classes):
        assert self.__class__.multi_class_flag or not self.__class__.class_count_flag
        super().__init__(world_generator)
        self.num_classes = num_classes

    def get_classes(self, world):  # iterable of classes
        raise NotImplementedError

    def generate(self, n, mode=None, noise=True, include_model=False):
        worlds = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
        if include_model:
            world_models = [None] * n
        classification = np.zeros(shape=(n, self.num_classes), dtype=np.float32)
        for i in range(n):
            world = self.world_generator(mode)
            if include_model:
                world_models[i] = world.model()
            worlds[i] = world.get_world(noise=noise)
            for c in self.get_classes(world):
                classification[i][c] += 1.0
        assert np.all(...)
        assert np.any(...)
        generated = dict()
        generated['world'] = worlds
        if include_model:
            generated['world-model'] = world_models
        generated['classification'] = classification
        return generated


class CaptionAgreementDataset(Dataset):

    value_types = {'world': 'world', 'world-model': 'model', 'caption': 'text', 'caption-length': 'int', 'agreement': 'float'}

    def __init__(self, world_generator, world_captioner, incorrect_world_ratio=0.5, correct_ratio=0.5, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None):
        super().__init__(world_generator)
        self.world_captioner = world_captioner
        self.incorrect_world_ratio = incorrect_world_ratio
        self.correct_ratio = correct_ratio
        self.train_correct_ratio = correct_ratio if train_correct_ratio is None else train_correct_ratio
        self.validation_correct_ratio = correct_ratio if validation_correct_ratio is None else validation_correct_ratio
        self.test_correct_ratio = correct_ratio if test_correct_ratio is None else test_correct_ratio

    @property
    def text_size(self):
        return self.world_captioner.caption_size

    @property
    def word_ids(self):
        return self.world_captioner.word_ids

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

    def generate(self, n, mode=None, noise=True, include_model=False):
        if mode == 'train':
            correct_ratio = self.train_correct_ratio
        elif mode == 'validation':
            correct_ratio = self.validation_correct_ratio
        elif mode == 'test':
            correct_ratio = self.test_correct_ratio
        else:
            correct_ratio = self.correct_ratio
        worlds = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
        if include_model:
            world_models = [None] * n
        captions = np.zeros(shape=(n, self.text_size), dtype=np.int32)
        caption_lengths = np.zeros(shape=(n, 1), dtype=np.int32)
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
            if include_model:
                world_models[i] = world.model()
            worlds[i] = world.get_world(noise=noise)
            caption_list.append(caption)
        caption_list = self.world_captioner.realize(caption_list)
        for i, caption in enumerate(caption_list):
            assert len(caption) <= self.text_size
            for j, word in enumerate(caption):
                captions[i][j] = self.word_ids[word]
            caption_lengths[i][0] = len(caption)
        generated = dict()
        generated['world'] = worlds
        if include_model:
            generated['world-model'] = world_models
        generated['caption'] = captions
        generated['caption-length'] = caption_lengths
        generated['agreement'] = agreements
        return generated


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

    def generate(self, n, mode=None, noise=True, include_model=False):
        if mode == 'train':
            distribution = self.train_distribution
        elif mode == 'validation':
            distribution = self.validation_distribution
        elif mode == 'test':
            distribution = self.test_distribution
        else:
            distribution = self.distribution
        worlds = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
        if include_model:
            world_models = [None] * n
        captions = np.zeros(shape=(n, self.text_size), dtype=np.int32)
        caption_lengths = np.zeros(shape=(n, 1), dtype=np.int32)
        agreements = np.zeros(shape=(n, 1), dtype=np.float32)
        for i in range(n):
            pick = random()
            cumulative = 0.0
            for dataset, prob in zip(self.datasets, distribution):
                cumulative += prob
                if pick < cumulative:
                    break
            generated = self.datasets[-1].generate(n=1, mode=mode, noise=noise, include_model=include_model)
            worlds[i] = generated['world'][0]
            if include_model:
                world_models[i] = generated['world-model'][0]
            captions[i] = generated['caption'][0]
            caption_lengths[i] = generated['caption-length'][0]
            agreements[i] = generated['agreement'][0]
        generated = dict()
        generated['world'] = worlds
        if include_model:
            generated['world-model'] = world_models
        generated['caption'] = captions
        generated['caption-length'] = caption_lengths
        generated['agreement'] = agreements


class ComparisonDataset(Dataset):

    value_types = {'reference': 'world', 'reference-model': 'model', 'comparison': 'world', 'comparison-model': 'model', 'agreement': 'float'}
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

    def generate(self, n, mode=None, noise=True, include_model=False):
        if mode == 'train':
            correct_ratio = self.train_correct_ratio
        elif mode == 'validation':
            correct_ratio = self.validation_correct_ratio
        elif mode == 'test':
            correct_ratio = self.test_correct_ratio
        else:
            correct_ratio = self.correct_ratio
        references = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
        comparisons = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
        if include_model:
            reference_models = [None] * n
            comparison_models = [None] * n
        agreements = np.zeros(shape=(n, 1), dtype=np.float32)
        for i in range(n):
            reference = self.world_generator(mode)
            if include_model:
                reference_models[i] = reference.model()
            references[i] = reference.get_world(noise=noise)
            if random() < correct_ratio:
                comparison = self.generate_correct_comparison(reference)
                agreements[i][0] = 1.0
            else:
                comparison = self.generate_incorrect_comparison(reference)
            if include_model:
                comparison_models[i] = comparison.model()
            comparisons[i] = comparison.get_world(noise=noise)
        generated = dict()
        generated['reference'] = references
        generated['comparison'] = comparisons
        if include_model:
            generated['reference-model'] = reference_models
            generated['comparison-model'] = comparison_models
        generated['agreement'] = agreements
        return generated


class CommunicationDataset(Dataset):

    value_types = {'alternative1': 'world', 'alternative1-model': 'model', 'alternative2': 'world', 'alternative2-model': 'model', 'reference': 'int'}
    valid_alternative_flag = None

    def __init__(self, world_generator, num_alternatives=2):
        super().__init__(world_generator)
        assert self.__class__.valid_alternative_flag is not None
        assert isinstance(num_alternatives, int) and num_alternatives >= 2
        self.num_alternatives = num_alternatives

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

    def generate(self, n, mode=None, noise=True, include_model=False):
        alternatives = [np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32) for _ in range(self.num_alternatives)]
        if include_model:
            alternative_models = [[None] * n for _ in range(self.num_alternatives)]
        references = np.zeros(shape=(n, 1), dtype=np.int32)
        for i in range(n):
            reference = randrange(self.num_alternatives)
            references[i] = reference
            world = self.world_generator(mode)
            if include_model:
                alternative_models[reference][i] = world.model()
            alternatives[reference][i] = world.get_world(noise=noise)
            for alt in range(self.num_alternatives):
                if alt == reference:
                    continue
                alternative = self.generate_alternative(world, mode)
                if include_model:
                    alternative_models[alt][i] = alternative.model()
                alternatives[alt][i] = alternative.get_world(noise=noise)
        generated = dict()
        generated['reference'] = references
        for alt in range(self.num_alternatives):
            alt_str = 'alternative' + str(alt + 1)
            generated[alt_str] = alternatives[alt]
            if include_model:
                generated[alt_str + '-model'] = alternative_models[alt]
        return generated
