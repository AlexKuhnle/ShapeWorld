from io import BytesIO
from importlib import import_module
import json
import os
from random import random, randrange
import numpy as np
from PIL import Image
from shapeworld.util import Archive
from shapeworld.world import World


class Dataset(object):

    dataset_name = None
    dataset_type = None
    value_types = {'world': 'world', 'world-model': 'model'}
    default_config = None

    def __init__(self, world_generator, vector_sizes=None, text_size=None, word_ids=None):
        assert self.__class__.dataset_name
        assert self.__class__.dataset_type
        assert all(value_type in ('int', 'float', 'vector', 'text', 'world', 'model') for value_type in self.__class__.value_types.values())
        self.world_generator = world_generator
        self.vector_sizes = vector_sizes
        self.text_size = text_size
        self.word_ids = word_ids

    @staticmethod
    def from_config(config=None, dataset_type=None, dataset_name=None, dataset_class=None):
        # explain type = 'load', 'mixer', possibilities, e.g. with ',', or with ';'?
        assert config is None or isinstance(config, dict) or isinstance(config, str)
        assert dataset_type is None or isinstance(dataset_type, str)
        assert dataset_name is None or isinstance(dataset_name, str)
        assert dataset_class is None or issubclass(dataset_class, Dataset)
        load = dataset_name == 'load'
        mix = dataset_name == 'mix'
        assert not load or not mix
        if config is not None:
            if isinstance(config, str):
                if mix and not os.path.isfile(config):
                    return MixerDataset(datasets=config.split(','))
                if os.path.isdir(config):  # load or nothing
                    load = True
                    config = os.path.join(config, 'specification.json')
                assert os.path.isfile(config)
                directory = os.path.dirname(config)
                with open(config, 'r') as filehandle:
                    config = json.load(fp=filehandle)
                if load and 'directory' not in config:
                    config['directory'] = directory
        if load:
            dataset = LoadedDataset(specification=config)
            assert dataset_type is None or dataset_type == dataset.type
            return dataset
        if mix:
            dataset = MixerDataset(**config)
            assert dataset_type is None or dataset_type == dataset.type
            return dataset
        if config is not None:
            if 'type' in config:
                if dataset_type is None:
                    dataset_type = config['type']
                else:
                    assert dataset_type == config['type']
            if 'name' in config:
                if dataset_name is None:
                    dataset_name = config['name']
                else:
                    assert dataset_name == config['name']
        assert dataset_type and dataset_name
        module = import_module('shapeworld.datasets.{}.{}'.format(dataset_type, dataset_name))
        dataset_class = module.dataset
        if config is None:
            assert dataset_class.default_config is not None
            config = dataset_class.default_config
        else:
            for key, value in dataset_class.default_config.items():
                if key not in config:
                    config[key] = value
        dataset = dataset_class(**config)
        return dataset

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
        return self.__class__.value_types

    @property
    def world_size(self):
        return self.world_generator.world_size

    def specification(self):
        specification = {'dataset_type': self.type, 'dataset_name': self.name, 'value_types': self.values, 'world_size': self.world_size}
        if self.vector_sizes:
            specification['vector_sizes'] = self.vector_sizes
        if self.text_size:
            specification['text_size'] = self.text_size
        if self.word_ids:
            specification['word_ids'] = self.word_ids
        return specification

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

    def zero_batch(self, n, include_model=False):
        batch = dict()
        for value_name, value_type in self.values.items():
            if value_type == 'int':
                batch[value_name] = np.zeros(shape=(n, 1), dtype=np.int32)
            elif value_type == 'float':
                batch[value_name] = np.zeros(shape=(n, 1), dtype=np.float32)
            elif value_type == 'vector':
                batch[value_name] = np.zeros(shape=(n, self.vector_sizes[value_name]), dtype=np.float32)
            elif value_type == 'text':
                batch[value_name] = np.zeros(shape=(n, self.text_size), dtype=np.int32)
            elif value_type == 'world':
                batch[value_name] = np.zeros(shape=(n, self.world_size, self.world_size, 3), dtype=np.float32)
            elif value_type == 'model' and include_model:
                batch[value_name] = [None] * n
        return batch

    def generate(self, n, mode=None, noise=True, include_model=False):  # mode: None, 'train', 'validation', 'test'
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            world = self.world_generator(mode)
            batch['world'][i] = world.get_world(noise=noise)
            if include_model:
                batch['world-model'][i] = world.model()
        return batch

    def iterate(self, n, mode=None, noise=True, include_model=False):
        while True:
            yield self.generate(n=n, mode=mode, noise=noise, include_model=include_model)

    def collect_captioner_statistics(self, filehandle, append=False):
        pass

    def close_captioner_statistics(self):
        pass

    def serialize_data(self, directory, generated, predicted=None, additional=None, archive=None, tiff=False):
        assert not additional or all(name not in self.values for name in additional)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        id2word = [word for word, _ in sorted(self.word_ids.items(), key=(lambda kv: kv[1]))] if self.word_ids else None
        temp_path = os.path.join(directory, 'temp')

        with Archive(directory=directory, mode='w', archive=archive) as write_file:
            for name in generated:
                Dataset.serialize_value(value=generated[name], value_name=name, value_type=self.values[name], write_file=write_file, id2word=id2word, tiff=tiff, temp_path=temp_path)
            if predicted:
                for name in predicted:
                    Dataset.serialize_value(value=predicted[name], value_name='predicted_' + name, value_type=self.values[name], write_file=write_file, id2word=id2word, tiff=tiff, temp_path=temp_path)
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
            value = '\n'.join(' '.join(id2word[word_id] for word_id in text if word_id) for text in value) + '\n'
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


class LoadedDataset(Dataset):

    dataset_name = 'Loaded'
    dataset_type = 'loaded'

    def __init__(self, specification):
        super().__init__(None, vector_sizes=specification.get('vector_sizes'), text_size=specification.get('text_size'), word_ids=specification.get('word_ids'))
        # assert per_part or not part_once
        self._dataset_name = specification['dataset_name']
        self._dataset_type = specification['dataset_type']
        self._value_types = specification['value_types']
        self._world_size = specification['world_size']
        self.world_model = specification.get('world_model')
        self.noise_range = specification.get('noise_range')
        self.tiff = specification.get('tiff')
        self.archive = specification.get('archive')
        self.per_part = True
        self.part_once = False

        self.parts = dict()
        directory = specification['directory']
        for root, dirs, files in os.walk(directory):
            if root == directory:
                assert len(files) == 1 and files[0] == 'specification.json'
                assert len(dirs) == 3 and 'train' in dirs and 'validation' in dirs and 'test' in dirs
            elif root[len(directory) + 1:] in ('train', 'validation', 'test'):
                mode = root[len(directory) + 1:]
                assert bool(dirs) != bool(files)
                if dirs:
                    assert all(d[:4] == 'part' and d[4:].isdigit() for d in dirs)
                    self.parts[mode] = [os.path.join(root, d) for d in dirs]
                else:
                    self.parts[mode] = [root]
        assert self.parts
        self.mode = None
        self.loaded = {value_name: [] for value_name, value_type in self.values.items() if value_type != 'model' or self.world_model}
        self.num_instances = 0

    @property
    def type(self):
        return self._dataset_type

    @property
    def name(self):
        return self._dataset_name

    @property
    def values(self):
        return self._value_types

    @property
    def world_size(self):
        return self._world_size

    def generate(self, n, mode=None, noise=True, include_model=False):
        assert noise or self.noise_range
        assert not include_model or self.world_model
        if not self.per_part:
            self.mode = None if mode else 'train'
        while self.mode != mode or self.num_instances < n:
            if self.mode != mode:
                self.mode = mode
                self.loaded = {value_name: [] for value_name, value_type in self.values.items() if value_type != 'model' or self.world_model}
            parts = self.parts[mode]
            part = randrange(len(parts))
            part_directory = parts.pop(part) if self.part_once else parts[part]
            self.num_instances = 0
            with Archive(directory=part_directory, mode='r', archive=self.archive) as read_file:
                for value_name, value in self.loaded.items():
                    value.extend(Dataset.deserialize_value(value_name=value_name, value_type=self.values[value_name], read_file=read_file, word2id=self.word_ids, tiff=self.tiff))
                    if self.num_instances:
                        assert len(value) == self.num_instances
                    else:
                        self.num_instances = len(value)
            temp_path = os.path.join(part_directory, 'temp')
            if os.path.isfile(temp_path):
                os.remove(temp_path)

        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            index = randrange(self.num_instances)
            self.num_instances -= 1
            for value_name, value_type in self.values.items():
                if value_type != 'model' or include_model:
                    value = self.loaded[value_name].pop(index)
                    if value_type == 'text':
                        batch[value_name][i][:len(value)] = value
                    else:
                        batch[value_name][i] = value
        if noise and self.noise_range:
            for value_name, value_type in self.loaded.items():
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


class MixerDataset(Dataset):

    dataset_name = 'Mixer'
    dataset_type = 'mixer'

    # accepts Dataset, config, str
    def __init__(self, datasets, consistent_batches=False, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        assert len(datasets) >= 1
        for n, dataset in enumerate(datasets):
            if not isinstance(dataset, Dataset):
                datasets[n] = Dataset.from_config(config=dataset)
        assert all(dataset.type == datasets[0].type for dataset in datasets)
        assert all(dataset.values == datasets[0].values for dataset in datasets)
        assert all(dataset.vector_sizes == datasets[0].vector_sizes for dataset in datasets)
        assert all(dataset.world_size == datasets[0].world_size for dataset in datasets)
        assert distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in distribution)
        assert train_distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in train_distribution)
        assert validation_distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in validation_distribution)
        assert test_distribution is None or all(isinstance(prob, float) or isinstance(prob, int) for prob in test_distribution)
        # combine text_size and word_ids information
        text_size = max(dataset.text_size for dataset in datasets)
        words = set(word for dataset in datasets for word in dataset.word_ids)
        words = sorted(words)
        word_ids = {words[n]: n for n in range(len(words))}
        super().__init__(None, vector_sizes=datasets[0].vector_sizes, text_size=text_size, word_ids=word_ids)
        for dataset in datasets:
            dataset.text_size = text_size
            dataset.word_ids = word_ids
        self.datasets = datasets
        self.consistent_batches = consistent_batches
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
        if mode == 'train':
            distribution = self.train_distribution
        elif mode == 'validation':
            distribution = self.validation_distribution
        elif mode == 'test':
            distribution = self.test_distribution
        else:
            distribution = self.distribution
        if self.consistent_batches:
            pick = random()
            cumulative = 0.0
            for dataset, prob in zip(self.datasets, distribution):
                cumulative += prob
                if pick < cumulative:
                    break
            return dataset.generate(n=n, mode=mode, noise=noise, include_model=include_model)
        else:
            batch = self.zero_batch(n, include_model=include_model)
            for i in range(n):
                pick = random()
                cumulative = 0.0
                for dataset, prob in zip(self.datasets, distribution):
                    cumulative += prob
                    if pick < cumulative:
                        break
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
    value_types = {'world': 'world', 'world-model': 'model', 'caption': 'text', 'caption-length': 'int', 'agreement': 'float'}

    def __init__(self, world_generator, world_captioner, caption_size, words, incorrect_world_ratio=None, correct_ratio=None, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None):
        assert isinstance(caption_size, int) and caption_size > 0
        assert isinstance(words, list) and len(words) > 0
        words = sorted(words)
        word_ids = {words[n]: n + 1 for n in range(len(words))}
        word_ids[''] = 0
        super().__init__(world_generator, text_size=caption_size, word_ids=word_ids)
        self.world_captioner = world_captioner
        self.incorrect_world_ratio = incorrect_world_ratio if incorrect_world_ratio is not None else 0.5
        self.correct_ratio = correct_ratio if correct_ratio is not None else 0.5
        self.train_correct_ratio = train_correct_ratio if train_correct_ratio is not None else self.correct_ratio
        self.validation_correct_ratio = validation_correct_ratio if validation_correct_ratio is not None else self.correct_ratio
        self.test_correct_ratio = test_correct_ratio if test_correct_ratio is not None else self.correct_ratio

    def generate_incorrect_world(self, caption, mode):
        if mode != 'train':
            mode = None
        while True:
            world = self.world_generator(mode)
            world_model = world.model()
            if caption.agreement(world_model) == 0.0:
                return world, world_model

    def generate_incorrect_caption(self, world, mode):
        if mode != 'train':
            mode = None
        while True:
            caption = self.world_captioner(world, False, mode)
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
        batch = self.zero_batch(n, include_model=include_model)
        captions = []
        for i in range(n):
            world = self.world_generator(mode)
            world_model = world.model()
            r = random()
            if r < correct_ratio:
                caption = self.world_captioner(world=world_model, correct=True, mode=mode)
                assert caption.agreement(world=world_model) == 1.0
                batch['agreement'][i][0] = 1.0
            else:
                r -= correct_ratio
                r /= 1.0 - correct_ratio
                if r < self.incorrect_world_ratio:
                    caption = self.world_captioner(world=world_model, correct=True, mode=mode)
                    world, world_model = self.generate_incorrect_world(caption=caption, mode=mode)
                else:
                    caption = self.generate_incorrect_caption(world=world_model, mode=mode)
                assert caption.agreement(world=world_model) == 0.0
            batch['world'][i] = world.get_world(noise=noise)
            if include_model:
                batch['world-model'][i] = world_model
            captions.append(caption)
        captions = self.world_captioner.realize(captions=captions)
        for i, caption in enumerate(captions):
            assert len(caption) <= self.text_size
            for j, word in enumerate(caption):
                batch['caption'][i][j] = self.word_ids[word]
            batch['caption-length'][i][0] = len(caption)
        return batch

    def collect_captioner_statistics(self, filehandle, append=False):
        self.world_captioner.collect_statistics(filehandle=filehandle, append=append)

    def close_captioner_statistics(self):
        self.world_captioner.close_statistics()


class ClassificationDataset(Dataset):

    dataset_type = 'classification'
    value_types = {'world': 'world', 'world-model': 'model', 'classification': 'vector'}
    multi_class_flag = False
    class_count_flag = False

    def __init__(self, world_generator, num_classes):
        assert self.__class__.multi_class_flag or not self.__class__.class_count_flag
        super().__init__(world_generator, vector_sizes={'classification': num_classes})
        self.num_classes = num_classes

    def get_classes(self, world):  # iterable of classes
        raise NotImplementedError

    def generate(self, n, mode=None, noise=True, include_model=False):
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            world = self.world_generator(mode)
            batch['world'][i] = world.get_world(noise=noise)
            if include_model:
                batch['world-model'][i] = world.model()
            c = None
            for c in self.get_classes(world):
                if self.__class__.class_count_flag:
                    batch['classification'][i][c] += 1.0
                else:
                    batch['classification'][i][c] = 1.0
            if not self.__class__.multi_class_flag:
                assert c is not None
        return batch


class CommunicationDataset(Dataset):

    dataset_type = 'communication'
    value_types = {'alternative1': 'world', 'alternative1-model': 'model', 'alternative2': 'world', 'alternative2-model': 'model', 'reference': 'int'}
    valid_alternative_flag = None

    def __init__(self, world_generator):  # , num_alternatives=2):
        super().__init__(world_generator)
        assert self.__class__.valid_alternative_flag is not None
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
            batch['alternative'][reference][i] = world.get_world(noise=noise)
            if include_model:
                batch['alternative-model'][reference][i] = world.model()
            for alt in range(self.num_alternatives):
                if alt == reference:
                    continue
                alternative = self.generate_alternative(world, mode)
                alt_str = 'alternative' + str(alt + 1)
                batch[alt_str][i] = alternative.get_world(noise=noise)
                if include_model:
                    batch[alt_str + '-model'][alt][i] = alternative.model()
        return batch


class CompositionDataset(Dataset):

    dataset_type = 'composition'
    value_types = {'compound': 'world', 'compound-model': 'model', 'component1': 'world', 'component2': 'world', 'composition-type': 'vector'}

    def __init__(self, world_generator, num_types):
        super().__init__(world_generator, vector_sizes={'composition-type': num_types})

    def get_components(self, world):
        raise NotImplementedError

    def generate(self, n, mode=None, noise=True, include_model=False):
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            compound = self.world_generator(mode)
            batch['compound'][i] = compound.get_world(noise=noise)
            if include_model:
                batch['compound-model'][i] = compound.model()
            component1, component2, composition_type = self.get_components(world=compound)
            batch['component1'][i] = component1.get_world(noise=noise)
            batch['component2'][i] = component2.get_world(noise=noise)
            batch['composition-type'][i] = composition_type
        return batch


class ComparisonDataset(Dataset):

    dataset_type = 'comparison'
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
        batch = self.zero_batch(n, include_model=include_model)
        for i in range(n):
            reference = self.world_generator(mode)
            batch['reference'][i] = reference.get_world(noise=noise)
            if include_model:
                batch['reference-model'][i] = reference.model()
            if random() < correct_ratio:
                comparison = self.generate_correct_comparison(reference)
                batch['agreement'][i][0] = 1.0
            else:
                comparison = self.generate_incorrect_comparison(reference)
            batch['comparison'][i] = comparison.get_world(noise=noise)
            if include_model:
                batch['comparison-model'][i] = comparison.model()
        return batch
