from importlib import import_module
from io import BytesIO
import json
from math import ceil, sqrt
import os
from random import random, randrange
import numpy as np
from PIL import Image
from shapeworld import CaptionRealizer
from shapeworld.util import Archive, cumulative_distribution, sample
from shapeworld.world import World


def dataset(dtype=None, name=None, config=None):
    # explain type = 'load', 'mixer', possibilities, e.g. with ',', or with ';'?
    assert config is None or isinstance(config, dict) or isinstance(config, str)
    assert dtype is None or isinstance(dtype, str)
    assert name is None or isinstance(name, str)
    load = mix = False
    if config is not None and isinstance(config, str):
        if config[:5] == 'load(' and config[-1] == ')':
            load = True
            config = config[5:-1]
        elif config[:4] == 'mix(' and config[-1] == ')':
            mix = True
            config = config[4:-1]
        assert not load or not mix
        # mix default config when names list
        if mix and not os.path.isfile(config):
            return DatasetMixer(datasets=config.split(','))
        if load and os.path.isdir(config):
            assert dtype and name
            directory = os.path.join(config, dtype, name)
            config = os.path.join(config, '{}-{}.json'.format(dtype, name))
        else:
            assert os.path.isfile(config)
            directory = os.path.dirname(config)
        with open(config, 'r') as filehandle:
            config = json.load(fp=filehandle)
        if load and 'directory' not in config:
            config['directory'] = directory
    if load:
        dataset = LoadedDataset(specification=config)
        assert dtype is None or dtype == dataset.type
        return dataset
    if mix:
        dataset = DatasetMixer(**config)
        assert dtype is None or dtype == dataset.type
        return dataset
    if config is not None:
        if 'type' in config:
            if dtype is None:
                dtype = config['type']
            else:
                assert dtype == config['type']
        if 'name' in config:
            if name is None:
                name = config['name']
            else:
                assert name == config['name']
    assert dtype and name
    module = import_module('shapeworld.datasets.{}.{}'.format(dtype, name))
    dclass = module.dataset
    if config is None:
        assert dclass.default_config is not None
        config = dclass.default_config
    else:
        for key, value in dclass.default_config.items():
            if key not in config:
                config[key] = value
    dataset = dclass(**config)
    return dataset


class Dataset(object):

    MAX_ATTEMPTS = 10
    dataset_name = None
    dataset_type = None
    dataset_values = {'world': 'world', 'world_model': 'model'}
    default_config = None

    def __init__(self, world_size, vectors=None, words=None):
        assert self.__class__.dataset_name
        assert self.__class__.dataset_type
        assert all(value_type in ('int', 'float', 'vector(int)', 'vector(float)', 'world', 'text', 'model') for value_type in self.__class__.dataset_values.values())
        self.world_size = world_size
        self.vectors = vectors
        self.words = words

    def __str__(self):
        return '{}-{}'.format(self.type, self.name)

    @property
    def type(self):
        return self.__class__.dataset_type

    @property
    def name(self):
        return self.__class__.dataset_name

    @property
    def values(self):
        return self.__class__.dataset_values

    def specification(self):
        specification = {'type': self.type, 'name': self.name, 'values': self.values, 'world_size': self.world_size}
        if self.vectors:
            specification['vectors'] = self.vectors
        if self.words:
            specification['words'] = self.words
        return specification

    @property
    def world_shape(self):
        return (self.world_size, self.world_size, 3)

    def vector_shape(self, value_name):
        return (self.vectors.get(value_name),)

    @property
    def vocabulary_size(self):
        return len(self.words)

    @property
    def vocabulary(self):
        return list(self.words.keys())

    def zero_batch(self, n, include_model=False):
        batch = dict()
        for value_name, value_type in self.values.items():
            if value_type == 'int':
                batch[value_name] = np.zeros(shape=(n, 1), dtype=np.int32)
            elif value_type == 'float':
                batch[value_name] = np.zeros(shape=(n, 1), dtype=np.float32)
            elif value_type == 'vector(int)' or value_type == 'text':
                batch[value_name] = np.zeros(shape=(n, self.vectors[value_name]), dtype=np.int32)
            elif value_type == 'vector(float)':
                batch[value_name] = np.zeros(shape=(n, self.vectors[value_name]), dtype=np.float32)
            elif value_type == 'world':
                batch[value_name] = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
            elif value_type == 'model' and include_model:
                batch[value_name] = [None] * n
        return batch

    def generate(self, n, mode=None, noise=True, include_model=False):  # mode: None, 'train', 'validation', 'test'
        raise NotImplementedError()

    def iterate(self, n, mode=None, noise=True, include_model=False):
        while True:
            yield self.generate(n=n, mode=mode, noise=noise, include_model=include_model)

    def collect_captioner_statistics(self, path, append=False):
        pass

    def close_captioner_statistics(self):
        pass

    def serialize(self, path, generated, additional=None, filename=None, archive=None, concat_worlds=False):
        assert not additional or all(value_name not in self.values for value_name in additional)
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        id2word = [word for word, _ in sorted(self.words.items(), key=(lambda kv: kv[1]))] if self.words else None

        with Archive(path=path, mode='w', archive=archive) as write_file:
            for value_name, value in generated.items():
                Dataset.serialize_value(value=value, value_name=value_name, value_type=self.values[value_name], write_file=write_file, concat_worlds=concat_worlds, id2word=id2word)
            if additional:
                for value_name, (value, value_type) in additional.items():
                    Dataset.serialize_value(value=value, value_name=value_name, value_type=value_type, write_file=write_file, concat_worlds=concat_worlds, id2word=id2word)

    @staticmethod
    def serialize_value(value, value_name, value_type, write_file, concat_worlds=False, id2word=None):
        if value_type == 'int':
            value = '\n'.join(str(int(x)) for x in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'float':
            value = '\n'.join(str(float(x)) for x in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'vector(int)' or value_type == 'vector(float)':
            value = '\n'.join(','.join(str(x) for x in vector) for vector in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'text':
            assert id2word
            value = '\n'.join(' '.join(id2word[word_id] for word_id in text if word_id) for text in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'world':
            if concat_worlds:
                size = ceil(sqrt(len(value)))
                worlds = []
                for y in range(ceil(len(value) / size)):
                    if y < len(value) // size:
                        worlds.append(np.concatenate([value[y * size + x] for x in range(size)], axis=1))
                    else:
                        worlds.append(np.concatenate([value[y * size + x] for x in range(len(value) % size)] + [np.zeros_like(a=value[0]) for _ in range(-len(value) % size)], axis=1))
                worlds = np.concatenate(worlds, axis=0)
                image = World.get_image(world_array=worlds)
                image_bytes = BytesIO()
                image.save(image_bytes, format='bmp')
                write_file(value_name + '.bmp', image_bytes.getvalue(), binary=True)
                image_bytes.close()
            else:
                for n in range(len(value)):
                    image = World.get_image(world_array=value[n])
                    image_bytes = BytesIO()
                    image.save(image_bytes, format='bmp')
                    write_file('{}-{}.bmp'.format(value_name, n), image_bytes.getvalue(), binary=True)
                    image_bytes.close()
        elif value_type == 'model':
            value = json.dumps(value)
            write_file(value_name + '.json', value)

    @staticmethod
    def deserialize_value(value_name, value_type, read_file, num_concat_worlds=0, word2id=None):
        if value_type == 'int':
            value = read_file(value_name + '.txt')
            value = [int(x) for x in value.split()]
            return value
        elif value_type == 'float':
            value = read_file(value_name + '.txt')
            value = [float(x) for x in value.split()]
            return value
        elif value_type == 'vector(int)':
            value = read_file(value_name + '.txt')
            value = [[int(x) for x in vector.split(',')] for vector in value.split()]
            return value
        elif value_type == 'vector(float)':
            value = read_file(value_name + '.txt')
            value = [[float(x) for x in vector.split(',')] for vector in value.split()]
            return value
        elif value_type == 'text':
            assert word2id
            value = read_file(value_name + '.txt')
            value = [[word2id[word] for word in text.split(' ')] for text in value.split('\n')[:-1]]
            return value
        elif value_type == 'world':
            if num_concat_worlds:
                size = ceil(sqrt(num_concat_worlds))
                image_bytes = read_file(value_name + '.bmp', binary=True)
                assert image_bytes is not None
                image_bytes = BytesIO(image_bytes)
                image = Image.open(image_bytes)
                worlds = World.from_image(image)
                height = worlds.shape[0] // ceil(num_concat_worlds / size)
                assert worlds.shape[0] % ceil(num_concat_worlds / size) == 0
                width = worlds.shape[1] // size
                assert worlds.shape[1] % size == 0
                value = []
                for y in range(ceil(num_concat_worlds / size)):
                    for x in range(size if y < num_concat_worlds // size else num_concat_worlds % size):
                        value.append(worlds[y * height: (y + 1) * height, x * width: (x + 1) * width, :])
            else:
                value = []
                n = 0
                while True:
                    image_bytes = read_file('{}-{}.bmp'.format(value_name, n), binary=True)
                    if image_bytes is None:
                        break
                    image_bytes = BytesIO(image_bytes)
                    image = Image.open(image_bytes)
                    value.append(World.from_image(image))
                    n += 1
            return value
        elif value_type == 'model':
            value = read_file(value_name + '.json')
            value = json.loads(value)
            return value


class LoadedDataset(Dataset):

    dataset_name = 'loaded'
    dataset_type = 'loaded'

    def __init__(self, specification):
        super(LoadedDataset, self).__init__(world_size=specification.pop('world_size'), vectors=specification.pop('vectors', None), words=specification.pop('words', None))
        # assert per_part or not part_once
        self._type = specification.pop('type')
        self._name = specification.pop('name')
        self._values = specification.pop('values')
        self.archive = specification.pop('archive', None)
        self.include_model = specification.pop('include_model', False)
        self.noise_range = specification.pop('noise_range', None)
        self.num_concat_worlds = specification.pop('num_concat_worlds', 0)
        self.specification = specification
        self.per_part = True
        self.part_once = False

        self.parts = dict()
        directory = specification['directory']
        for root, dirs, files in os.walk(directory):
            if root == directory:
                assert not files
                assert len(dirs) <= 4 and 'train' in dirs and 'validation' in dirs and 'test' in dirs and (len(dirs) == 3 or 'tf-records' in dirs)
            elif root[len(directory) + 1:] in ('train', 'validation', 'test', 'tf-records'):
                mode = root[len(directory) + 1:]
                if dirs:
                    assert all(d[:4] == 'part' and d[4:].isdigit() for d in dirs)
                    assert all(f == 'captioner_statistics.csv' for f in files)
                    self.parts[mode] = [os.path.join(root, d) for d in dirs]
                else:
                    assert all(f == 'captioner_statistics.csv' or f[:4] == 'part' for f in files)
                    self.parts[mode] = [os.path.join(root, f) for f in files if f != 'captioner_statistics.csv']
        assert self.parts
        self.mode = None
        self.loaded = {value_name: [] for value_name, value_type in self.values.items() if value_type != 'model' or self.include_model}
        self.num_instances = 0

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def values(self):
        return self._values

    def __getattr__(self, name):
        if name in self.specification:
            return self.specification[name]
        raise AttributeError()

    def generate(self, n, mode=None, noise=True, include_model=False):
        assert noise or self.noise_range
        assert not include_model or self.include_model
        if not self.per_part:
            self.mode = None if mode else 'train'
        while self.mode != mode or self.num_instances < n:
            if self.mode != mode:
                self.mode = mode
                self.loaded = {value_name: [] for value_name, value_type in self.values.items() if value_type != 'model' or self.include_model}
            parts = self.parts[mode]
            part = randrange(len(parts))
            path = parts.pop(part) if self.part_once else parts[part]
            self.num_instances = 0
            with Archive(path=path, mode='r', archive=self.archive) as read_file:
                for value_name, value in self.loaded.items():
                    value.extend(Dataset.deserialize_value(value_name=value_name, value_type=self.values[value_name], read_file=read_file, num_concat_worlds=self.num_concat_worlds, word2id=self.words))
                    if self.num_instances:
                        assert len(value) == self.num_instances
                    else:
                        self.num_instances = len(value)

        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            index = randrange(self.num_instances)
            self.num_instances -= 1
            for value_name, value_type in self.values.items():
                if value_type == 'model' and not self.include_model:
                    continue
                value = self.loaded[value_name].pop(index)
                if value_type == 'text':
                    batch[value_name][i][:len(value)] = value
                elif value_type != 'model' or include_model:
                    batch[value_name][i] = value
        if noise and self.noise_range:
            for value_name, value_type in self.values.items():
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


class DatasetMixer(Dataset):

    dataset_name = 'mixer'
    dataset_type = 'mixer'

    # accepts Dataset, config, str
    def __init__(self, datasets, consistent_batches=False, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        assert len(datasets) >= 1
        for n, dataset in enumerate(datasets):
            if not isinstance(dataset, Dataset):
                datasets[n] = Dataset.dataset(config=dataset)
        assert all(dataset.type == datasets[0].type for dataset in datasets)
        assert all(dataset.values == datasets[0].values for dataset in datasets)
        assert all(dataset.world_size == datasets[0].world_size for dataset in datasets)
        assert all(sorted(dataset.vectors) == sorted(datasets[0].vectors) for dataset in datasets)
        assert all((dataset.words is None) == (datasets[0].words is None) for dataset in datasets)
        # combine vectors and words information
        vectors = {value_name: max(dataset.vectors[value_name] for dataset in datasets) for value_name in datasets[0].vectors}
        words = sorted(set(word for dataset in datasets for word in dataset.words))
        words = {words[n]: n for n in range(len(words))}
        super(DatasetMixer, self).__init__(None, vectors=vectors, words=words)
        for dataset in datasets:
            dataset.vectors = self.vectors
            dataset.words = self.words
        self.datasets = datasets
        self.consistent_batches = consistent_batches
        assert not distribution or len(distribution) == len(datasets)
        self.distribution = cumulative_distribution(distribution or [1] * len(datasets))
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(self.distribution)
        self.train_distribution = cumulative_distribution(train_distribution) if train_distribution else self.distribution
        self.validation_distribution = cumulative_distribution(validation_distribution) if validation_distribution else self.distribution
        self.test_distribution = cumulative_distribution(test_distribution) if test_distribution else self.distribution

    @property
    def type(self):
        return self.datasets[0].type

    @property
    def name(self):
        return 'mixer'

    @property
    def values(self):
        return self.datasets[0].values

    @property
    def world_size(self):
        return self.datasets[0].world_size

    def generate(self, n, mode=None, noise=True, include_model=False):
        if mode is None:
            distribution = self.distribution
        if mode == 'train':
            distribution = self.train_distribution
        elif mode == 'validation':
            distribution = self.validation_distribution
        elif mode == 'test':
            distribution = self.test_distribution
        if self.consistent_batches:
            dataset = sample(distribution, self.datasets)
            return dataset.generate(n=n, mode=mode, noise=noise, include_model=include_model)
        else:
            batch = self.zero_batch(n, include_model=include_model)
            for i in range(n):
                dataset = sample(distribution, self.datasets)
                generated = dataset.generate(n=1, mode=mode, noise=noise, include_model=include_model)
                for value_name, value_type in self.values.items():
                    value = generated[value_name][0]
                    if value_type == 'text':
                        batch[value_name][i][:len(value)] = value
                    else:
                        batch[value_name][i] = value
        return batch


class CaptionAgreementDataset(Dataset):

    dataset_type = 'agreement'
    dataset_values = {'world': 'world', 'world_model': 'model', 'caption': 'text', 'caption_model': 'model', 'caption_length': 'int', 'agreement': 'float'}

    def __init__(self, world_generator, world_captioner, caption_size, words, incorrect_world_ratio=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None, caption_realizer=None):
        assert isinstance(caption_size, int) and caption_size > 0
        assert isinstance(words, list) and len(words) > 0
        assert words == sorted(words)  # !!!
        words = sorted(words)
        words = {words[n]: n + 1 for n in range(len(words))}
        words[''] = 0
        super(CaptionAgreementDataset, self).__init__(world_size=world_generator.world_size, vectors={'caption': caption_size}, words=words)
        self.world_generator = world_generator
        self.world_captioner = world_captioner
        self.caption_size = caption_size
        self.incorrect_world_ratio = incorrect_world_ratio if incorrect_world_ratio is not None else 0.5
        self.correct_ratio = correct_ratio if correct_ratio is not None else 0.5
        self.train_correct_ratio = train_correct_ratio if train_correct_ratio is not None else self.correct_ratio
        self.validation_correct_ratio = validation_correct_ratio if validation_correct_ratio is not None else self.correct_ratio
        self.test_correct_ratio = test_correct_ratio if test_correct_ratio is not None else self.correct_ratio
        if isinstance(caption_realizer, CaptionRealizer):
            self.caption_realizer = caption_realizer
        else:
            assert caption_realizer is None or isinstance(caption_realizer, str)
            self.caption_realizer = CaptionRealizer.from_name(name=(caption_realizer or 'dmrs'))
        self.world_captioner.set_realizer(self.caption_realizer)

    def generate_incorrect_world(self, caption, mode):
        if mode != 'train':
            mode = None
        for _ in range(Dataset.MAX_ATTEMPTS):
            world = self.world_generator(mode)
            if not world:
                continue
            world_model = world.model()
            if caption.agreement(world_model) == 0.0:
                return world, world_model
        return None, None

    def generate_incorrect_caption(self, world, mode):
        if mode != 'train':
            mode = None
        for _ in range(Dataset.MAX_ATTEMPTS):
            caption = self.world_captioner(world, False, mode)
            if caption and caption.agreement(world) == 0.0:
                return caption
        return None

    def generate(self, n, mode=None, noise=True, include_model=False):
        if mode == 'train':
            correct_ratio = self.train_correct_ratio
        elif mode == 'validation':
            correct_ratio = self.validation_correct_ratio
        elif mode == 'test':
            correct_ratio = self.test_correct_ratio
        else:
            correct_ratio = self.correct_ratio
        batch = self.zero_batch(n, include_model=include_model)
        captions = [None] * n
        for i in range(n):
            r = random()
            correct = r < correct_ratio
            for _ in range(Dataset.MAX_ATTEMPTS):
                world = self.world_generator(mode)
                if not world:
                    continue
                world_model = world.model()
                if correct:
                    caption = self.world_captioner(world=world_model, correct=True, mode=mode)
                    if not caption:
                        continue
                    assert caption.agreement(world=world_model) == 1.0
                    batch['agreement'][i][0] = 1.0
                else:
                    if (r - correct_ratio) / (1.0 - correct_ratio) < self.incorrect_world_ratio:
                        caption = self.world_captioner(world=world_model, correct=True, mode=mode)
                        if not caption:
                            continue
                        world, world_model = self.generate_incorrect_world(caption=caption, mode=mode)
                        if not world:
                            continue
                    else:
                        caption = self.generate_incorrect_caption(world=world_model, mode=mode)
                        if not caption:
                            continue
                    assert caption.agreement(world=world_model) == 0.0
                batch['world'][i] = world.get_array(noise=noise)
                captions[i] = caption
                if include_model:
                    batch['world_model'][i] = world_model
                    batch['caption_model'][i] = caption.model()
                break
            else:
                raise Exception()
        captions = self.caption_realizer.realize(captions=captions)
        for i, caption in enumerate(captions):
            assert len(caption) <= self.caption_size
            for j, word in enumerate(caption):
                batch['caption'][i][j] = self.words[word]
            batch['caption_length'][i][0] = len(caption)
        return batch

    def collect_captioner_statistics(self, path, append=False):
        self.world_captioner.collect_statistics(path=path, append=append)

    def close_captioner_statistics(self):
        self.world_captioner.close_statistics()


class ClassificationDataset(Dataset):

    dataset_type = 'classification'
    dataset_values = {'world': 'world', 'world_model': 'model', 'classification': 'vector(float)'}

    def __init__(self, world_generator, num_classes, multi_class=False, class_count=False):
        super(ClassificationDataset, self).__init__(world_size=world_generator.world_size, vectors={'classification': num_classes})
        assert multi_class or not class_count
        self.world_generator = world_generator
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.class_count = class_count

    def specification(self):
        specification = super(ClassificationDataset, self).specification()
        specification['num_classes'] = self.num_classes
        specification['multi_class'] = self.multi_class
        specification['class_count'] = self.class_count
        return specification

    def get_classes(self, world):  # iterable of classes
        raise NotImplementedError

    def generate(self, n, mode=None, noise=True, include_model=False):
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            world = self.world_generator(mode)
            batch['world'][i] = world.get_array(noise=noise)
            if include_model:
                batch['world_model'][i] = world.model()
            c = None
            for c in self.get_classes(world):
                if self.class_count:
                    batch['classification'][i][c] += 1.0
                else:
                    batch['classification'][i][c] = 1.0
            if not self.multi_class:
                assert c is not None
        return batch


class CommunicationDataset(Dataset):

    dataset_type = 'communication'
    dataset_values = {'alternative1': 'world', 'alternative1_model': 'model', 'alternative2': 'world', 'alternative2_model': 'model', 'reference': 'int'}
    valid_alternative_flag = None

    def __init__(self, world_generator):  # , num_alternatives=2):
        super(CommunicationDataset, self).__init__(world_size=world_generator.world_size)
        assert self.__class__.valid_alternative_flag is not None
        self.world_generator = world_generator
        # assert isinstance(num_alternatives, int) and num_alternatives >= 2
        self.num_alternatives = 2

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
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            reference = randrange(self.num_alternatives)
            batch['reference'][i] = reference
            world = self.world_generator(mode)
            batch['alternative'][reference][i] = world.get_array(noise=noise)
            if include_model:
                batch['alternative_model'][reference][i] = world.model()
            for alt in range(self.num_alternatives):
                if alt == reference:
                    continue
                alternative = self.generate_alternative(world, mode)
                alt_str = 'alternative' + str(alt + 1)
                batch[alt_str][i] = alternative.get_array(noise=noise)
                if include_model:
                    batch[alt_str + '_model'][alt][i] = alternative.model()
        return batch


class CompositionDataset(Dataset):

    dataset_type = 'composition'
    dataset_values = {'compound': 'world', 'compound_model': 'model', 'component1': 'world', 'component2': 'world', 'composition_type': 'int'}

    def __init__(self, world_generator, num_types):
        super(CompositionDataset, self).__init__(world_size=world_generator.world_size, vectors={'composition-type': num_types})
        self.world_generator = world_generator

    def get_components(self, world):
        raise NotImplementedError

    def generate(self, n, mode=None, noise=True, include_model=False):
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            compound = self.world_generator(mode)
            batch['compound'][i] = compound.get_array(noise=noise)
            if include_model:
                batch['compound_model'][i] = compound.model()
            component1, component2, composition_type = self.get_components(world=compound)
            batch['component1'][i] = component1.get_array(noise=noise)
            batch['component2'][i] = component2.get_array(noise=noise)
            batch['composition_type'][i] = composition_type
        return batch


class ComparisonDataset(Dataset):

    dataset_type = 'comparison'
    values = {'reference': 'world', 'reference_model': 'model', 'comparison': 'world', 'comparison_model': 'model', 'agreement': 'float'}
    correct_comparison_flag = None
    incorrect_comparison_flag = None

    def __init__(self, world_generator, correct_ratio, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None):
        super(ComparisonDataset, self).__init__(world_size=world_generator.world_size)
        assert self.__class__.correct_comparison_flag is not None and self.__class__.incorrect_comparison_flag is not None
        self.world_generator = world_generator
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
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            reference = self.world_generator(mode)
            batch['reference'][i] = reference.get_array(noise=noise)
            if include_model:
                batch['reference_model'][i] = reference.model()
            if random() < correct_ratio:
                comparison = self.generate_correct_comparison(reference)
                batch['agreement'][i][0] = 1.0
            else:
                comparison = self.generate_incorrect_comparison(reference)
            batch['comparison'][i] = comparison.get_array(noise=noise)
            if include_model:
                batch['comparison_model'][i] = comparison.model()
        return batch
