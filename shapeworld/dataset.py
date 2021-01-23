from importlib import import_module
from io import BytesIO
import json
from math import ceil, sqrt
import os
from random import random, randrange
import numpy as np
from PIL import Image
from shapeworld import util


class Dataset(object):

    def __init__(self, values, world_size, pixel_noise_stddev=None, vectors=None, vocabularies=None, language=None):
        assert self.type and self.name
        assert all(value_name != 'alternatives' or value_type == 'int' for value_name, value_type in values.items())
        self.values = values
        if isinstance(world_size, int):
            self.world_size = world_size
        else:
            self.world_size = tuple(world_size)
        self.pixel_noise_stddev = pixel_noise_stddev
        self.vectors = {value_name: shape if isinstance(shape, int) else tuple(shape) for value_name, shape in vectors.items()}
        self.vocabularies = dict()
        if vocabularies is not None:
            for name, vocabulary in vocabularies.items():
                if isinstance(vocabulary, dict):
                    assert all(isinstance(word, str) and isinstance(index, int) for word, index in vocabulary.items())
                    assert sorted(vocabulary.values()) == list(range(len(vocabulary)))
                    self.vocabularies[name] = vocabulary
                else:
                    vocabulary = {word: index for index, word in enumerate((word for word in vocabulary if word != '' and word != '[UNKNOWN]'), 1)}
                    vocabulary[''] = 0
                    vocabulary['[UNKNOWN]'] = len(vocabulary)
                    self.vocabularies[name] = vocabulary
        self.language = language

    @staticmethod
    def create(dtype=None, name=None, variant=None, language=None, config=None, **kwargs):
        assert variant is None or name is not None
        assert language is None or name is not None

        if isinstance(name, str):
            try:
                name = json.loads(name)
            except Exception:
                pass

        if isinstance(variant, str):
            try:
                variant = json.loads(variant)
            except Exception:
                pass

        if isinstance(config, str):
            try:
                config = json.loads(config)
            except Exception:
                pass

        if isinstance(name, (tuple, list)):
            try:
                if not isinstance(variant, list):
                    variant = [variant for _ in name]
                if not isinstance(config, list):
                    config = [config for _ in name]
                datasets = list()
                for n, v, c in zip(name, variant, config):
                    # for v in vs:
                    datasets.append(Dataset.create(dtype=dtype, name=n, variant=v, language=language, config=c))
                dataset = DatasetMixer(datasets=datasets, **kwargs)
                assert dtype == dataset.type
                assert language is None or language == dataset.language
                return dataset
            except TypeError:
                assert False

        if isinstance(variant, (tuple, list)):
            try:
                if not isinstance(config, list):
                    config = [config for _ in variant]
                datasets = list()
                for v, c in zip(variant, config):
                    datasets.append(Dataset.create(dtype=dtype, name=name, variant=v, language=language, config=c))
                dataset = DatasetMixer(datasets=datasets, **kwargs)
                assert dtype == dataset.type
                assert name == dataset.name
                assert language is None or language == dataset.language
                return dataset
            except TypeError:
                assert False

        if isinstance(config, (tuple, list)):
            assert len(kwargs) == 0
            try:
                datasets = list()
                for c in config:
                    # if isinstance(c, dict):
                    #     c = dict(c)
                    # datasets.append(c)
                    datasets.append(Dataset.create(dtype=dtype, name=name, variant=variant, language=language, config=c))
                dataset = DatasetMixer(datasets=datasets, **kwargs)
                assert dtype is None or dtype == dataset.type
                assert language is None or language == dataset.language
                return dataset
            except TypeError:
                assert False

        if config is None:
            config = dict()

        elif isinstance(config, dict):
            config = dict(config)

        elif os.path.isdir(config):
            assert dtype is not None and name is not None
            full_name = name
            if variant is not None:
                full_name = '{}-{}'.format(full_name, variant)
            if language is not None:
                full_name = '{}-{}'.format(full_name, language)
            directory = config
            config = os.path.join(config, '{}-{}.json'.format(dtype, full_name))
            with open(config, 'r') as filehandle:
                config = json.load(fp=filehandle)
            if 'directory' not in config:
                config['directory'] = directory
            return Dataset.create(dtype=dtype, name=name, variant=variant, language=language, config=config, **kwargs)

        elif os.path.isfile(config):
            with open(config, 'r') as filehandle:
                config = json.load(fp=filehandle)
            d = config.pop('type', None)
            if dtype is None:
                dtype = d
            else:
                assert dtype == d
            n = config.pop('name', None)
            if name is None:
                name = n
            else:
                assert name == n
            v = config.pop('variant', None)
            if variant is None:
                variant = v
            else:
                assert variant == v
            l = config.pop('language', language)
            if language is None:
                language = l
            else:
                assert language == l
            if 'config' in config:
                assert not kwargs
                kwargs = config
                config = kwargs.pop('config')
            return Dataset.create(dtype=dtype, name=name, variant=variant, language=language, config=config, **kwargs)

        else:
            raise Exception('Invalid config value: ' + str(config))

        if config.pop('generated', False):
            assert dtype is None or 'type' not in config or config['type'] == dtype
            assert name is None or 'name' not in config or config['name'] == name
            assert variant is None or 'variant' not in config or config['variant'] == variant
            assert language is None or 'language' not in config or config['language'] == language
            if 'dtype' in config:
                assert dtype == config['type']
                dtype = config['type']
            else:
                assert dtype is not None
                config['type'] = dtype
            if 'name' in config:
                assert name == config['name']
                name = config['name']
            else:
                assert name is not None
                config['name'] = name
            if 'variant' in config:
                assert variant == config['variant']
                variant = config.get('variant')
            elif variant is not None:
                config['variant'] = variant
            if 'language' in config:
                assert language == config['language']
                language = config.get('language')
            elif language is not None:
                config['language'] = language
            dataset = LoadedDataset(specification=config, **kwargs)
            assert dtype == dataset.type
            assert name == dataset.name
            assert variant is None or variant == dataset.variant
            assert language is None or language == dataset.language
            return dataset

        else:
            assert variant is None
            config.pop('directory', None)

            for key, value in kwargs.items():
                assert key not in config
                config[key] = value

            if dtype is None:
                dtype = config.pop('type')
            else:
                dtype_config = config.pop('type', dtype)
                assert dtype_config == dtype
            if name is None:
                name = config.pop('name')
            else:
                name_config = config.pop('name', name)
                assert name_config == name
            if 'language' in config:
                assert language is None or config['language'] == language
            elif language is not None:
                config['language'] = language

            module = import_module('shapeworld.datasets.{}.{}'.format(dtype, name))
            class_name = util.class_name(name) + 'Dataset'
            for key, module in module.__dict__.items():
                if key == class_name:
                    break
            dataset = module(**config)
            assert dtype == dataset.type
            assert name == dataset.name
            return dataset

    def __str__(self):
        if self.language is None:
            return '{} {}'.format(self.type, self.name)
        else:
            return '{} {} ({})'.format(self.type, self.name, self.language)

    @property
    def type(self):
        raise NotImplementedError

    @property
    def name(self):
        name = self.__class__.__name__
        assert name[-7:] == 'Dataset'
        return util.real_name(name[:-7])

    def specification(self):
        specification = dict(type=self.type, name=self.name, values=self.values)
        if isinstance(self.world_size, int):
            specification['world_size'] = self.world_size
        else:
            specification['world_size'] = list(self.world_size)
        if self.vectors:
            specification['vectors'] = self.vectors
        if self.vocabularies:
            specification['vocabularies'] = self.vocabularies
        if self.language:
            specification['language'] = self.language
        return specification

    def world_shape(self):
        if isinstance(self.world_size, int):
            return (self.world_size, self.world_size, 3)
        else:
            return (self.world_size[0], self.world_size[1], 3)

    def vector_shape(self, value_name):
        shape = self.vectors.get(value_name)
        if isinstance(shape, int):
            return (self.vectors.get(value_name),)
        else:
            return shape

    def vocabulary_size(self, value_type):
        if self.vocabularies is None or value_type not in self.vocabularies:
            return -1
        else:
            return len(self.vocabularies[value_type])

    def vocabulary(self, value_type):
        if self.vocabularies is None or value_type not in self.vocabularies:
            return None
        else:
            return [word for word, _ in sorted(self.vocabularies[value_type].items(), key=(lambda kv: kv[1]))]

    def to_surface(self, value_type, word_ids):
        id2word = self.vocabulary(value_type)
        assert id2word is not None
        if word_ids.ndim == 1:
            return ' '.join(id2word[word_id] for word_id in word_ids)
        elif word_ids.ndim == 2:
            return [self.to_surface(value_type, word_ids) for word_ids in word_ids]
        else:
            assert False

    def from_surface(self, value_type, words):
        word2id = self.vocabularies.get(value_type)
        assert word2id is not None
        if isinstance(words, str):
            return np.asarray(word2id[word] for word in words.split(' '))
        elif isinstance(words, list):
            if len(words) > 0 and ' ' in words[0]:
                return [self.from_surface(value_type, words) for words in words]
            else:
                return np.asarray(word2id[word] for word in words)
        else:
            assert False

    def apply_pixel_noise(self, world):
        if self.pixel_noise_stddev is not None and self.pixel_noise_stddev > 0.0:
            noise = np.random.normal(loc=0.0, scale=self.pixel_noise_stddev, size=world.shape)
            mask = (noise < -2.0 * self.pixel_noise_stddev) + (noise > 2.0 * self.pixel_noise_stddev)
            while np.any(a=mask):
                noise -= mask * noise
                noise += mask * np.random.normal(loc=0.0, scale=self.pixel_noise_stddev, size=world.shape)
                mask = (noise < -2.0 * self.pixel_noise_stddev) + (noise > 2.0 * self.pixel_noise_stddev)
            world += noise
            np.clip(world, a_min=0.0, a_max=1.0, out=world)
        return world

    def zero_batch(self, n, include_model=False, alternatives=False):
        batch = dict()
        for value_name, value_type in self.values.items():
            value_type, alts = util.alternatives_type(value_type=value_type)
            if alternatives and alts:
                if value_type == 'int':
                    batch[value_name] = [[] for _ in range(n)]
                elif value_type == 'float':
                    batch[value_name] = [[] for _ in range(n)]
                elif value_type == 'vector(int)' or value_type in self.vocabularies:
                    batch[value_name] = [[np.zeros(shape=self.vector_shape(value_name), dtype=np.int32)] for _ in range(n)]
                elif value_type == 'vector(float)':
                    batch[value_name] = [[np.zeros(shape=self.vector_shape(value_name), dtype=np.float32)] for _ in range(n)]
                elif value_type == 'world':
                    batch[value_name] = [[np.zeros(shape=self.world_shape(), dtype=np.float32)] for _ in range(n)]
                elif value_type == 'model' and include_model:
                    batch[value_name] = [[] for _ in range(n)]
            else:
                if value_type == 'int' and (value_name != 'alternatives' or alternatives):
                    batch[value_name] = np.zeros(shape=(n,), dtype=np.int32)
                elif value_type == 'float':
                    batch[value_name] = np.zeros(shape=(n,), dtype=np.float32)
                elif value_type == 'vector(int)' or value_type in self.vocabularies:
                    batch[value_name] = np.zeros(shape=((n,) + self.vector_shape(value_name)), dtype=np.int32)
                elif value_type == 'vector(float)':
                    batch[value_name] = np.zeros(shape=((n,) + self.vector_shape(value_name)), dtype=np.float32)
                elif value_type == 'world':
                    batch[value_name] = np.zeros(shape=((n,) + self.world_shape()), dtype=np.float32)
                elif value_type == 'model' and include_model:
                    batch[value_name] = [None] * n
        return batch

    def generate(self, n, mode=None, include_model=False, alternatives=False):  # mode: None, 'train', 'validation', 'test'
        raise NotImplementedError

    def iterate(self, n, mode=None, include_model=False, alternatives=False, iterations=None):
        i = 0
        while iterations is None or i < iterations:
            yield self.generate(n=n, mode=mode, include_model=include_model, alternatives=alternatives)
            i += 1

    def get_html(self, generated, image_format='bmp', image_dir=''):
        return None

    def serialize(self, path, generated, additional=None, filename=None, archive=None, html=False, numpy_formats=(), image_format='bmp', concat_worlds=False):
        assert not additional or all(value_name not in self.values for value_name in additional)
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with util.Archive(path=path, mode='w', archive=archive) as write_file:
            for value_name, value in generated.items():
                self.serialize_value(
                    path=path,
                    value=value,
                    value_name=value_name,
                    write_file=write_file,
                    numpy_format=(value_name in numpy_formats),
                    image_format=image_format,
                    concat_worlds=concat_worlds
                )
            if additional:
                for value_name, (value, value_type) in additional.items():
                    self.serialize_value(
                        path=path,
                        value=value,
                        value_name=value_name,
                        write_file=write_file,
                        value_type=value_type,
                        numpy_format=(value_name in numpy_formats),
                        image_format=image_format,
                        concat_worlds=concat_worlds
                    )
            if html:
                html = self.get_html(generated=generated, image_format=image_format)
                assert html is not None
                write_file(filename='data.html', value=html)

    def serialize_value(self, path, value, value_name, write_file, value_type=None, numpy_format=False, image_format='bmp', concat_worlds=False):
        if value_type is None:
            value_type = self.values[value_name]
        value_type, alts = util.alternatives_type(value_type=value_type)
        if value_name == 'alternatives':
            assert value_type == 'int'
            assert not numpy_format
            value = '\n'.join(str(int(x)) for x in value) + '\n'
            write_file('alternatives.txt', value)
        elif value_type == 'int':
            assert not numpy_format
            if alts:
                value = '\n'.join(';'.join(str(x)for x in xs) for xs in value) + '\n'
            else:
                value = '\n'.join(str(x) for x in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'float':
            assert not numpy_format
            if alts:
                value = '\n'.join(';'.join(str(round(x, 3))for x in xs) for xs in value) + '\n'
            else:
                value = '\n'.join(str(round(x, 3)) for x in value) + '\n'
            write_file(value_name + '.txt', value)
        elif value_type == 'vector(int)':
            if numpy_format:
                np.save(path + '-' + value_name + '.npy', value)
            elif alts:
                value = '\n'.join(';'.join(','.join(str(x) for x in vector.flatten()) for vector in vectors) for vectors in value) + '\n'
                write_file(value_name + '.txt', value)
            else:
                value = '\n'.join(','.join(str(x) for x in vector.flatten()) for vector in value) + '\n'
                write_file(value_name + '.txt', value)
        elif value_type == 'vector(float)':
            if numpy_format:
                np.save(path + '-' + value_name + '.npy', value)
            elif alts:
                value = '\n'.join(';'.join(','.join(str(round(x, 3)) for x in vector.flatten()) for vector in vectors) for vectors in value) + '\n'
                write_file(value_name + '.txt', value)
            else:
                value = '\n'.join(','.join(str(round(x, 3)) for x in vector.flatten()) for vector in value) + '\n'
                write_file(value_name + '.txt', value)
        elif value_type == 'world':
            from shapeworld.world import World
            if numpy_format:
                np.save(path + '-' + value_name + '.npy', value)
            elif concat_worlds:
                assert not alts
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
                image.save(image_bytes, format=image_format)
                write_file('{}.{}'.format(value_name, image_format), image_bytes.getvalue(), binary=True)
                image_bytes.close()
            else:
                for n in range(len(value)):
                    if alts:
                        for i, v in enumerate(value[n]):
                            image = World.get_image(world_array=v)
                            image_bytes = BytesIO()
                            image.save(image_bytes, format=image_format)
                            write_file('{}-{}-{}.{}'.format(value_name, n, i, image_format), image_bytes.getvalue(), binary=True)
                            image_bytes.close()
                    else:
                        image = World.get_image(world_array=value[n])
                        image_bytes = BytesIO()
                        image.save(image_bytes, format=image_format)
                        write_file('{}-{}.{}'.format(value_name, n, image_format), image_bytes.getvalue(), binary=True)
                        image_bytes.close()
        elif value_type == 'model':
            assert not numpy_format
            value = json.dumps(obj=value, indent=2, sort_keys=True)
            write_file(value_name + '.json', value)
        else:
            assert not numpy_format
            id2word = self.vocabulary(value_type=value_type)
            if alts:
                value = '\n\n'.join('\n'.join(' '.join(id2word[word_id] for word_id in words if word_id) for words in words_alts) for words_alts in value) + '\n\n'
            else:
                value = '\n'.join(' '.join(id2word[word_id] for word_id in words if word_id) for words in value) + '\n'
            write_file(value_name + '.txt', value)

    def deserialize_value(self, path, value_name, read_file, value_type=None, numpy_format=False, image_format='bmp', num_concat_worlds=0):
        if value_type is None:
            value_type = self.values[value_name]
        value_type, alts = util.alternatives_type(value_type=value_type)
        if value_name == 'alternatives':
            assert value_type == 'int'
            assert not numpy_format
            value = read_file('alternatives.txt')
            return [int(x) for x in value.split('\n')[:-1]]
        elif value_type == 'int':
            assert not numpy_format
            value = read_file(value_name + '.txt')
            if alts:
                value = [[int(x) for x in xs.split(';')] for xs in value.split('\n')[:-1]]
            else:
                value = [int(x) for x in value.split('\n')[:-1]]
            return value
        elif value_type == 'float':
            assert not numpy_format
            value = read_file(value_name + '.txt')
            if alts:
                value = [[float(x) for x in xs.split(';')] for xs in value.split('\n')[:-1]]
            else:
                value = [float(x) for x in value.split('\n')[:-1]]
            return value
        elif value_type == 'vector(int)':
            if numpy_format:
                path, extension = os.path.splitext(path)
                while extension != '':
                    path, extension = os.path.splitext(path)
                value = np.load(path + '-' + value_name + '.npy')
            else:
                value = read_file(value_name + '.txt')
                shape = self.vector_shape(value_name=value_name)
                if alts:
                    value = [[np.array(object=[int(x) for x in vector.split(',')], dtype=np.int32).reshape(shape) for vector in vectors.split(';')] for vectors in value.split('\n')[:-1]]
                else:
                    value = [np.array(object=[int(x) for x in vector.split(',')], dtype=np.int32).reshape(shape) for vector in value.split('\n')[:-1]]
            return value
        elif value_type == 'vector(float)':
            if numpy_format:
                path, extension = os.path.splitext(path)
                while extension != '':
                    path, extension = os.path.splitext(path)
                value = np.load(path + '-' + value_name + '.npy')
            else:
                value = read_file(value_name + '.txt')
                shape = self.vector_shape(value_name=value_name)
                if alts:
                    value = [[np.array(object=[float(x) for x in vector.split(',')], dtype=np.float32).reshape(shape) for vector in vectors.split(';')] for vectors in value.split('\n')[:-1]]
                else:
                    value = [np.array(object=[0.0 if 'e' in x else float(x) for x in vector.split(',')], dtype=np.float32).reshape(shape) for vector in value.split('\n')[:-1]]
            return value
        elif value_type == 'world':
            from shapeworld.world import World
            if numpy_format:
                path, extension = os.path.splitext(path)
                while extension != '':
                    path, extension = os.path.splitext(path)
                value = np.load(path + '-' + value_name + '.npy')
            elif num_concat_worlds:
                assert not alts
                size = ceil(sqrt(num_concat_worlds))
                image_bytes = read_file('{}.{}'.format(value_name, image_format), binary=True)
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
                value = list()
                n = 0
                flag = True
                while flag:
                    if alts:
                        i = 0
                        v = list()
                        while True:
                            image_bytes = read_file('{}-{}-{}.{}'.format(value_name, n, i, image_format), binary=True)
                            if image_bytes is None:
                                flag = False
                                break
                            image_bytes = BytesIO(image_bytes)
                            image = Image.open(image_bytes)
                            v.append(World.from_image(image))
                            i += 1
                        value.append(v)
                    else:
                        image_bytes = read_file('{}-{}.{}'.format(value_name, n, image_format), binary=True)
                        if image_bytes is None:
                            break
                        image_bytes = BytesIO(image_bytes)
                        image = Image.open(image_bytes)
                        value.append(World.from_image(image))
                    n += 1
            return value
        elif value_type == 'model':
            assert not numpy_format
            value = read_file(value_name + '.json')
            value = json.loads(s=value)
            return value
        else:
            assert not numpy_format
            word2id = self.vocabularies.get(value_type)
            value = read_file(value_name + '.txt')
            if alts:
                value = [[[word2id[word] for word in words.split(' ')] for words in words_alts.split('\n')] for words_alts in value.split('\n\n')[:-1]]
            else:
                value = [[word2id[word] for word in words.split(' ')] for words in value.split('\n')[:-1]]
            return value


class LoadedDataset(Dataset):

    def __init__(self, specification, random_sampling=True, pixel_noise_stddev=None, exclude_values=()):
        self._type = specification.pop('type')
        self._name = specification.pop('name')
        self.variant = specification.pop('variant', None)
        self.directory = specification.pop('directory')
        relative_directory = specification.get('relative_directory')
        if relative_directory is not None:
            self.directory = os.path.join(self.directory, relative_directory)
        self.archive = specification.pop('archive', None)
        self.include_model = specification.pop('include_model', False)
        self.numpy_formats = tuple(specification.pop('numpy_formats', ()))
        self.image_format = specification.pop('image_format', 'bmp')
        self.num_concat_worlds = specification.pop('num_concat_worlds', 0)
        self._specification = specification
        self.random_sampling = random_sampling

        values = specification.pop('values')
        for value in exclude_values:
            values.pop(value, None)

        if pixel_noise_stddev is None:
            pixel_noise_stddev = specification.pop('pixel_noise_stddev', None)
        else:
            assert 'pixel_noise_stddev' not in specification

        super(LoadedDataset, self).__init__(values=values, world_size=specification.pop('world_size'), pixel_noise_stddev=pixel_noise_stddev, vectors=specification.pop('vectors', None), vocabularies=specification.pop('vocabularies', None), language=specification.pop('language', None))

        self.shards = None
        self.records_shards = None
        for root, dirs, files in os.walk(self.directory):
            if root == self.directory:
                dirs = sorted(d for d in dirs if d[0] != '.' and not d.startswith('temp-'))
                files = sorted(f for f in files if f[0] != '.' and not f.endswith('.npy'))
                if len(dirs) == 0:
                    assert all(f[:5] == 'shard' or f[:4] == 'part' for f in files)
                    if any(f[-13:] != '.tfrecords.gz' for f in files):
                        self.shards = [os.path.join(root, f) for f in files if f[-13:] != '.tfrecords.gz']
                    if any(f[-13:] == '.tfrecords.gz' for f in files):
                        self.records_shards = [os.path.join(root, f) for f in files if f[-13:] == '.tfrecords.gz']
                elif set(dirs) <= {'train', 'validation', 'test'}:
                    assert len(files) == 0
                    self.shards = dict()
                    self.records_shards = dict()
                else:
                    assert all(d[:5] == 'shard' or d[:4] == 'part' for d in dirs)
                    self.shards = [os.path.join(root, d) for d in dirs]
                    if len(files) > 0:
                        assert all((f[:5] == 'shard' or f[:4] == 'part') and f[-13:] == '.tfrecords.gz' for f in files)
                        self.records_shards = [os.path.join(root, f) for f in files]
            elif root[len(self.directory) + 1:] in ('train', 'validation', 'test'):
                dirs = sorted(d for d in dirs if d[0] != '.' and not d.startswith('temp-'))
                files = sorted(f for f in files if f[0] != '.' and not f.endswith('.npy'))
                mode = root[len(self.directory) + 1:]
                if len(dirs) > 0:
                    assert all(d[:5] == 'shard' or d[:4] == 'part' for d in dirs)
                    self.shards[mode] = [os.path.join(root, d) for d in dirs]
                    if files:
                        assert all((f[:5] == 'shard' or f[:4] == 'part') and f[-13:] == '.tfrecords.gz' for f in files)
                        self.records_shards[mode] = [os.path.join(root, f) for f in files]
                else:
                    assert all(f[:5] == 'shard' or f[:4] == 'part' for f in files)
                    if any(f[-13:] != '.tfrecords.gz' for f in files):
                        self.shards[mode] = [os.path.join(root, f) for f in files if f[-13:] != '.tfrecords.gz']
                    if any(f[-13:] == '.tfrecords.gz' for f in files):
                        self.records_shards[mode] = [os.path.join(root, f) for f in files if f[-13:] == '.tfrecords.gz']

        self.loaded = dict()
        self.shard = dict()
        self.num_instances = dict()
        self.num_alternatives = dict()

    def __str__(self):
        name = '{} {}'.format(self.type, self.name)
        if self.variant is not None:
            name += '-{}'.format(self.variant)
        if self.language is not None:
            name += ' ({})'.format(self.language)
        return name

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def specification(self):
        specification = super(LoadedDataset, self).specification()
        specification.update(self._specification)
        return specification

    def __getattr__(self, name):
        try:
            return super(LoadedDataset, self).__getattr__(name=name)
        except AttributeError:
            if name in self._specification:
                return self._specification[name]
            else:
                raise

    def get_records_paths(self, mode):
        if mode == 'none':
            mode = None
        assert (mode is None) != isinstance(self.records_shards, dict)
        assert (mode is None and self.records_shards is not None) or mode in self.records_shards
        if mode is None:
            return self.records_shards
        else:
            assert mode in self.records_shards
            return self.records_shards[mode]

    def generate(self, n, mode=None, include_model=False, alternatives=False):
        if mode == 'none':
            mode = None
        assert (mode is None) != isinstance(self.shards, dict)
        assert (mode is None and self.shards is not None) or mode in self.shards
        assert not include_model or self.include_model

        if mode is None:
            mode_shards = self.shards
        else:
            mode_shards = self.shards[mode]

        if mode not in self.loaded:
            self.loaded[mode] = dict()
            self.shard[mode] = -1
            self.num_instances[mode] = 0
            self.num_alternatives[mode] = 0
            for value_name, value_type in self.values.items():
                value_type, alts = util.alternatives_type(value_type=value_type)
                if value_type != 'model' or self.include_model:
                    self.loaded[mode][value_name] = list()

        while (self.num_instances[mode] < n) if (self.random_sampling or alternatives) else (self.num_alternatives[mode] < n):
            if self.random_sampling:
                next_shard = self.shard[mode]
                while len(mode_shards) > 1 and next_shard == self.shard[mode]:
                    next_shard = randrange(len(mode_shards))
                self.shard[mode] = next_shard
            else:
                self.shard[mode] = (self.shard[mode] + 1) % len(mode_shards)
            self.num_instances[mode] = 0
            with util.Archive(path=mode_shards[self.shard[mode]], mode='r', archive=self.archive) as read_file:
                for value_name, value in self.loaded[mode].items():
                    value.extend(self.deserialize_value(
                        path=mode_shards[self.shard[mode]],
                        value_name=value_name,
                        read_file=read_file,
                        numpy_format=(value_name in self.numpy_formats),
                        image_format=self.image_format,
                        num_concat_worlds=self.num_concat_worlds
                    ))
                    if self.num_instances[mode] == 0:
                        self.num_instances[mode] = self.num_alternatives[mode] = len(value)
                    else:
                        assert len(value) == self.num_instances[mode]
                    if value_name == 'alternatives':
                        self.num_alternatives[mode] = sum(value)

        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)

        for i in range(n):
            if self.random_sampling:
                index = randrange(self.num_instances[mode])
            else:
                index = 0
            if 'alternatives' not in self.values:
                alt_index = -1
                self.num_instances[mode] -= 1
                self.num_alternatives[mode] -= 1
            elif alternatives or self.loaded[mode]['alternatives'][index] == 1:
                alt_index = -1
                self.num_instances[mode] -= 1
                self.num_alternatives[mode] -= self.loaded[mode]['alternatives'][index]
            elif self.random_sampling:
                alt_index = randrange(self.loaded[mode]['alternatives'][index])
                self.num_instances[mode] -= 1
                self.num_alternatives[mode] -= self.loaded[mode]['alternatives'][index]
            else:
                alt_index = 0
                self.loaded[mode]['alternatives'][index] -= 1
                self.num_alternatives[mode] -= 1

            for value_name, value_type in self.values.items():
                value_type, alts = util.alternatives_type(value_type=value_type)
                if value_type == 'model' and not self.include_model:
                    continue
                if self.random_sampling or alt_index == -1:
                    value = self.loaded[mode][value_name].pop(index)
                else:
                    value = self.loaded[mode][value_name][index]
                if value_type == 'model' and not include_model:
                    continue
                if not alternatives and alts:
                    value = value.pop(alt_index)
                if not alternatives and value_name == 'alternatives':
                    continue
                if value_type in self.vocabularies:
                    batch[value_name][i][:len(value)] = value
                else:
                    batch[value_name][i] = value

        for value_name, value_type in self.values.items():
            value_type, _ = util.alternatives_type(value_type=value_type)
            if value_type == 'world':
                batch[value_name] = self.apply_pixel_noise(world=batch[value_name])

        return batch

    def epoch(self, n, mode=None, include_model=False, alternatives=False):
        if mode == 'none':
            mode = None
        assert (mode is None) != isinstance(self.shards, dict)
        assert (mode is None and self.shards is not None) or mode in self.shards
        assert not include_model or self.include_model

        if mode is None:
            mode_shards = self.shards
        else:
            mode_shards = self.shards[mode]
        available_shards = list(range(len(mode_shards)))
        loaded = dict()
        for value_name, value_type in self.values.items():
            value_type, alts = util.alternatives_type(value_type=value_type)
            if value_type != 'model' or self.include_model:
                loaded[value_name] = list()
        num_instances = 0

        while available_shards:
            if self.random_sampling:
                shard = available_shards.pop(randrange(len(available_shards)))
            else:
                shard = available_shards.pop(0)
            num_instances = 0
            with util.Archive(path=mode_shards[shard], mode='r', archive=self.archive) as read_file:
                for value_name, value in loaded.items():
                    value.extend(self.deserialize_value(
                        path=mode_shards[shard],
                        value_name=value_name,
                        read_file=read_file,
                        numpy_format=(value_name in self.numpy_formats),
                        image_format=self.image_format,
                        num_concat_worlds=self.num_concat_worlds
                    ))
                    if num_instances == 0:
                        num_instances = num_alternatives = len(value)
                    else:
                        assert len(value) == num_instances
                    if value_name == 'alternatives':
                        num_alternatives = sum(value)

            while (num_instances >= n) if alternatives else (num_alternatives >= n):
                batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)

                for i in range(n):
                    if self.random_sampling:
                        index = randrange(num_instances)
                    else:
                        index = 0
                    if 'alternatives' not in self.values:
                        alt_index = -1
                        num_instances -= 1
                        num_alternatives -= 1
                    elif alternatives or loaded['alternatives'][index] == 1:
                        alt_index = -1
                        num_instances -= 1
                        num_alternatives -= loaded['alternatives'][index]
                    else:
                        if self.random_sampling:
                            alt_index = randrange(loaded['alternatives'][index])
                        else:
                            alt_index = 0
                        loaded['alternatives'][index] -= 1
                        num_alternatives -= 1

                    for value_name, value_type in self.values.items():
                        value_type, alts = util.alternatives_type(value_type=value_type)
                        if value_type == 'model' and not self.include_model:
                            continue
                        if alt_index == -1:
                            value = loaded[value_name].pop(index)
                            if not alternatives and alts:
                                value = value.pop(0)
                        elif alts:
                            value = loaded[value_name][index].pop(alt_index)
                        else:
                            value = loaded[value_name][index]
                        if value_type == 'model' and not include_model:
                            continue
                        if not alternatives and value_name == 'alternatives':
                            continue
                        if value_type in self.vocabularies:
                            batch[value_name][i][:len(value)] = value
                        else:
                            batch[value_name][i] = value

                for value_name, value_type in self.values.items():
                    value_type, _ = util.alternatives_type(value_type=value_type)
                    if value_type == 'world':
                        batch[value_name] = self.apply_pixel_noise(world=batch[value_name])

                yield batch

                if not available_shards:
                    if alternatives:
                        if 0 < num_instances < n:
                            n = num_instances

                    else:
                        if 0 < num_alternatives < n:
                            n = num_alternatives

        assert num_instances == 0 and num_alternatives == 0

    def get_html(self, generated, image_format='bmp', image_dir=''):
        module = import_module('shapeworld.datasets.{}.{}'.format(self.type, self.name))
        class_name = util.class_name(self.name) + 'Dataset'
        for key, dclass in module.__dict__.items():
            if key == class_name:
                break
        return dclass.get_html(self, generated=generated, image_format=image_format, image_dir=image_dir)


class DatasetMixer(Dataset):

    # accepts Dataset, config, str
    def __init__(self, datasets, consistent_batches=False, distribution=None, train_distribution=None, validation_distribution=None, test_distribution=None):
        assert len(datasets) >= 1
        self.datasets = list()
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                dataset = Dataset.create(config=dataset)
            self.datasets.append(dataset)
        assert all(dataset.type == self.datasets[0].type for dataset in self.datasets)
        assert all(dataset.language == self.datasets[0].language for dataset in self.datasets)
        assert all(dataset.values == self.datasets[0].values for dataset in self.datasets)
        assert all(dataset.world_size == self.datasets[0].world_size for dataset in self.datasets)
        assert all(sorted(dataset.vectors) == sorted(self.datasets[0].vectors) for dataset in self.datasets)
        assert all(sorted(dataset.vocabularies) == sorted(self.datasets[0].vocabularies) for dataset in self.datasets)
        # combine vectors and words information
        values = self.datasets[0].values
        world_size = self.datasets[0].world_size
        vectors = {value_name: max(dataset.vectors[value_name] for dataset in self.datasets) for value_name in self.datasets[0].vectors}
        vocabularies = dict()
        for name in self.datasets[0].vocabularies:
            vocabularies[name] = sorted(set(word for dataset in self.datasets for word in dataset.vocabularies[name]))
        language = self.datasets[0].language
        super(DatasetMixer, self).__init__(values=values, world_size=world_size, vectors=vectors, vocabularies=vocabularies, language=language)
        self.translations = list()
        for dataset in self.datasets:
            dataset.vectors = self.vectors
            dataset.vocabularies = self.vocabularies
            if isinstance(dataset, LoadedDataset):
                translation = dict()
                for name, vocabulary in dataset.vocabularies.items():
                    translation[name] = np.vectorize({index: self.vocabularies[name][word] for word, index in vocabulary.items()}.__getitem__)
                self.translations.append(translation)
            else:
                self.translations.append(None)
        self.consistent_batches = consistent_batches
        assert not distribution or len(distribution) == len(self.datasets)
        distribution = util.value_or_default(distribution, [1] * len(self.datasets))
        self.distribution = util.cumulative_distribution(distribution)
        assert bool(train_distribution) == bool(validation_distribution) == bool(test_distribution)
        assert not train_distribution or len(train_distribution) == len(validation_distribution) == len(test_distribution) == len(self.distribution)
        self.train_distribution = util.cumulative_distribution(util.value_or_default(train_distribution, distribution))
        self.validation_distribution = util.cumulative_distribution(util.value_or_default(validation_distribution, distribution))
        self.test_distribution = util.cumulative_distribution(util.value_or_default(test_distribution, distribution))

    @property
    def type(self):
        return self.datasets[0].type

    @property
    def name(self):
        return '+'.join(dataset.name for dataset in self.datasets)

    def generate(self, n, mode=None, include_model=False, alternatives=False):
        if mode == 'none':
            mode = None
        if mode is None:
            distribution = self.distribution
        if mode == 'train':
            distribution = self.train_distribution
        elif mode == 'validation':
            distribution = self.validation_distribution
        elif mode == 'test':
            distribution = self.test_distribution
        if self.consistent_batches:
            dataset = util.sample(distribution, self.datasets)
            return dataset.generate(n=n, mode=mode, include_model=include_model, alternatives=alternatives)
        else:
            batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
            for i in range(n):
                sample = util.sample(distribution)
                generated = self.datasets[sample].generate(n=1, mode=mode, include_model=include_model, alternatives=alternatives)
                for value_name, value in generated.items():
                    value_type = self.values[value_name]
                    if value_type in self.vocabularies:
                        batch[value_name][i][:value.shape[1]] = value[0]
                    else:
                        batch[value_name][i] = value[0]
        return batch


class ClassificationDataset(Dataset):

    def __init__(self, world_generator, num_classes, multi_class=False, count_class=False, pixel_noise_stddev=None):
        values = dict(world='world', world_model='model', classification='vector(float)')
        vectors = dict(classification=num_classes)
        super(ClassificationDataset, self).__init__(values=values, world_size=world_generator.world_size, vectors=vectors, pixel_noise_stddev=pixel_noise_stddev)
        assert multi_class or not count_class
        self.world_generator = world_generator
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.count_class = count_class

    @property
    def type(self):
        return 'classification'

    def specification(self):
        specification = super(ClassificationDataset, self).specification()
        specification['num_classes'] = self.num_classes
        specification['multi_class'] = self.multi_class
        specification['count_class'] = self.count_class
        return specification

    def get_classes(self, world):  # iterable of classes
        raise NotImplementedError

    def generate(self, n, mode=None, include_model=False, alternatives=False):
        if mode == 'none':
            mode = None

        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
        for i in range(n):

            while not self.world_generator.initialize(mode=mode):
                pass

            while True:
                world = self.world_generator()
                if world is not None:
                    break

            batch['world'][i] = self.apply_pixel_noise(world=world.get_array(world_array=batch['world'][i]))

            if include_model:
                batch['world_model'][i] = world.model()

            c = None
            for c in self.get_classes(world):
                if self.count_class:
                    batch['classification'][i][c] += 1.0
                else:
                    batch['classification'][i][c] = 1.0

            if not self.multi_class:
                assert c is not None

        return batch

    def get_html(self, generated, image_format='bmp', image_dir=''):
        classifications = generated['classification']
        data_html = list()
        for n, classification in enumerate(classifications):
            data_html.append('<div class="instance"><div class="world"><img src="{image_dir}world-{world}.{format}" alt="world-{world}.{format}"></div><div class="num"><p><b>({num})</b></p></div><div class="classification"><p>'.format(image_dir=image_dir, world=n, format=image_format, num=(n + 1)))
            comma = False
            for c, count in enumerate(classification):
                if count == 0.0:
                    continue
                if comma:
                    data_html.append(',&ensp;')
                else:
                    comma = True
                if self.count_class:
                    data_html.append('{count:.0f} &times; class {c}'.format(c=c, count=count))
                else:
                    data_html.append('class {c}'.format(c=c))
            data_html.append('</p></div></div>')
        html = '<!DOCTYPE html><html><head><title>{dtype} {name}</title><style>.data{{width: 100%; height: 100%;}} .instance{{width: 100%; display: flex; margin-top: 1px; margin-bottom: 1px; background-color: #DDEEFF; vertical-align: middle; align-items: center;}} .world{{height: {world_height}px; display: inline-block; flex-grow: 0; vertical-align: middle;}} .num{{width: 50px; display: inline-block; flex-grow: 0; text-align: center; vertical-align: middle; margin-left: 10px;}} .classification{{display: inline-block; flex-grow: 1; vertical-align: middle; margin-left: 10px;}}</style></head><body><div class="data">{data}</div></body></html>'.format(
            dtype=self.type,
            name=self.name,
            world_height=self.world_shape()[0],
            data=''.join(data_html)
        )
        return html


class CaptionAgreementDataset(Dataset):

    GENERATOR_INIT_FREQUENCY = 25
    CAPTIONER_INIT_FREQUENCY = 100
    CAPTIONER_INIT_FREQUENCY2 = 5

    def __init__(self, world_generator, world_captioner, caption_size, vocabulary, pixel_noise_stddev=None, caption_realizer='dmrs', language=None, worlds_per_instance=1, captions_per_instance=1, correct_ratio=0.5, train_correct_ratio=None, validation_correct_ratio=None, test_correct_ratio=None):
        if worlds_per_instance > 1 or captions_per_instance > 1:
            values = dict(agreement='alternatives(float)')
        else:
            values = dict(agreement='float')
        if worlds_per_instance > 1:
            values.update(world='alternatives(world)', world_model='alternatives(model)', alternatives='int')
        else:
            values.update(world='world', world_model='model')
        if captions_per_instance > 1:
            values.update(caption='alternatives(language)', caption_length='alternatives(int)', caption_pn='alternatives(pn)', caption_pn_length='alternatives(int)', caption_rpn='alternatives(pn)', caption_rpn_length='alternatives(int)', caption_model='alternatives(model)', alternatives='int')
        else:
            values.update(caption='language', caption_length='int', caption_pn='pn', caption_pn_length='int', caption_rpn='pn', caption_rpn_length='int', caption_model='model')
        assert isinstance(caption_size, int) and caption_size > 0
        vocabulary = list(vocabulary)
        assert len(vocabulary) > 0 and vocabulary == sorted(vocabulary), sorted(vocabulary)  # [(w1, w2) for w1, w2 in zip(vocabulary, sorted(vocabulary)) if w1 != w2]
        self.world_generator = world_generator
        self.world_captioner = world_captioner
        from shapeworld.realizers import CaptionRealizer
        if isinstance(caption_realizer, CaptionRealizer):
            self.caption_realizer = caption_realizer
        else:
            assert caption_realizer is None or isinstance(caption_realizer, str)
            self.caption_realizer = CaptionRealizer.from_name(
                name=caption_realizer,
                language=util.value_or_default(language, 'english')
            )
        self.world_captioner.set_realizer(self.caption_realizer)
        vectors = dict(
            caption=caption_size,
            caption_pn=self.world_captioner.pn_length(),
            caption_rpn=self.world_captioner.pn_length()
        )
        vocabularies = dict(
            language=vocabulary,
            pn=sorted(self.world_captioner.pn_symbols())
        )
        super(CaptionAgreementDataset, self).__init__(
            values=values,
            world_size=world_generator.world_size,
            pixel_noise_stddev=pixel_noise_stddev,
            vectors=vectors,
            vocabularies=vocabularies,
            language=language
        )
        assert worlds_per_instance == 1 or captions_per_instance == 1
        self.worlds_per_instance = worlds_per_instance
        self.captions_per_instance = captions_per_instance
        self.correct_ratio = correct_ratio
        self.train_correct_ratio = util.value_or_default(train_correct_ratio, self.correct_ratio)
        self.validation_correct_ratio = util.value_or_default(validation_correct_ratio, self.correct_ratio)
        self.test_correct_ratio = util.value_or_default(test_correct_ratio, self.correct_ratio)
        self.pn_arity = self.world_captioner.pn_arity()
        self.pn_arity[''] = 1
        self.pn_arity['[UNKNOWN]'] = 1

    @property
    def type(self):
        return 'agreement'

    def specification(self):
        specification = super(CaptionAgreementDataset, self).specification()
        specification['worlds_per_instance'] = self.worlds_per_instance
        specification['captions_per_instance'] = self.captions_per_instance
        specification['pn_arity'] = self.pn_arity
        return specification

    def generate(self, n, mode=None, include_model=False, alternatives=False):
        if mode == 'none':
            mode = None
        if mode == 'train':
            correct_ratio = self.train_correct_ratio
        elif mode == 'validation':
            correct_ratio = self.validation_correct_ratio
        elif mode == 'test':
            correct_ratio = self.test_correct_ratio
        else:
            correct_ratio = self.correct_ratio

        pn2id = self.vocabularies['pn']
        unknown = pn2id['[UNKNOWN]']
        pn_size = self.vector_shape('caption_pn')[0]

        batch = self.zero_batch(n, include_model=include_model, alternatives=alternatives)
        captions = list()
        for i in range(n):
            correct = random() < correct_ratio
            # print(i, correct, flush=True)
            # print(i, correct, end=', ', flush=True)

            resample = 0
            while True:

                if resample % self.__class__.GENERATOR_INIT_FREQUENCY == 0:
                    if resample // self.__class__.GENERATOR_INIT_FREQUENCY >= 1:
                        # print(i, 'world')
                        pass
                    while not self.world_generator.initialize(mode=mode):
                        pass
                    # print(self.world_generator.model())

                if resample % self.__class__.CAPTIONER_INIT_FREQUENCY == 0:
                    if resample // self.__class__.CAPTIONER_INIT_FREQUENCY >= 1:
                        # print(i, 'caption')
                        # print(i, resample, 'caption', correct, self.world_captioner.model())
                        # assert False
                        pass
                    if self.worlds_per_instance > 1:
                        correct = True
                        while not self.world_captioner.initialize(mode=mode, correct=False):
                            pass
                    else:
                        while not self.world_captioner.initialize(mode=mode, correct=correct):
                            pass
                        assert self.world_captioner.incorrect_possible()
                    # print(self.world_captioner.model(), flush=True)

                resample += 1

                world = self.world_generator()
                if world is None:
                    continue

                if self.worlds_per_instance > 1:
                    caption = self.world_captioner(world=world)
                    if caption is None:
                        continue
                    caption = self.world_captioner.get_correct_caption()
                else:
                    caption = self.world_captioner(world=world)
                # print('c', caption)

                if caption is not None:
                    break

            if alternatives and (self.worlds_per_instance > 1 or self.captions_per_instance > 1):
                batch['agreement'][i].append(float(correct))
            else:
                batch['agreement'][i] = float(correct)

            if alternatives and self.captions_per_instance > 1:
                batch['alternatives'][i] = self.captions_per_instance
                batch['caption'][i].extend(batch['caption'][i][0].copy() for _ in range(self.captions_per_instance - 1))
                batch['caption_pn'][i].extend(batch['caption_pn'][i][0].copy() for _ in range(self.captions_per_instance - 1))
                batch['caption_rpn'][i].extend(batch['caption_rpn'][i][0].copy() for _ in range(self.captions_per_instance - 1))
                captions.append(caption)
                pn = caption.polish_notation()
                assert len(pn) <= pn_size, (len(pn), pn_size, pn)
                for k, pn_symbol in enumerate(pn):
                    assert pn_symbol in pn2id, (pn_symbol, pn2id)
                    batch['caption_pn'][i][0][k] = pn2id.get(pn_symbol, unknown)
                batch['caption_pn_length'][i].append(len(pn))
                rpn = caption.polish_notation(reverse=True)
                assert len(rpn) <= pn_size, (len(rpn), pn_size, rpn)
                for k, pn_symbol in enumerate(rpn):
                    assert pn_symbol in pn2id, (pn_symbol, pn2id)
                    batch['caption_rpn'][i][0][k] = pn2id.get(pn_symbol, unknown)
                batch['caption_rpn_length'][i].append(len(rpn))
                if include_model:
                    batch['caption_model'][i].append(caption.model())
                for j in range(1, self.captions_per_instance):
                    correct = random() < correct_ratio
                    resample = 0
                    while True:
                        if resample % self.__class__.CAPTIONER_INIT_FREQUENCY2 == 0:
                            if resample // self.__class__.CAPTIONER_INIT_FREQUENCY2 >= 1:
                                # print(i, j, '2nd caption')
                                # print(i, 'caption', correct, self.world_captioner.model())
                                pass
                            while not self.world_captioner.initialize(mode=mode, correct=correct):
                                pass
                        resample += 1
                        caption = self.world_captioner(world=world)
                        if caption is not None:
                            break
                    captions.append(caption)
                    pn = caption.polish_notation()
                    assert len(pn) <= pn_size, (len(pn), pn_size, pn)
                    for k, pn_symbol in enumerate(pn):
                        assert pn_symbol in pn2id, (pn_symbol, pn2id)
                        batch['caption_pn'][i][j][k] = pn2id.get(pn_symbol, unknown)
                    batch['caption_pn_length'][i].append(len(pn))
                    rpn = caption.polish_notation(reverse=True)
                    assert len(rpn) <= pn_size, (len(rpn), pn_size, rpn)
                    for k, pn_symbol in enumerate(rpn):
                        assert pn_symbol in pn2id, (pn_symbol, pn2id)
                        batch['caption_rpn'][i][j][k] = pn2id.get(pn_symbol, unknown)
                    batch['caption_rpn_length'][i].append(len(rpn))
                    if include_model:
                        batch['caption_model'][i].append(caption.model())
                    batch['agreement'][i].append(float(correct))

            else:
                captions.append(caption)
                pn = caption.polish_notation()
                assert len(pn) <= pn_size, (len(pn), pn_size, pn)
                for k, pn_symbol in enumerate(pn):
                    assert pn_symbol in pn2id, (pn_symbol, pn2id)
                    batch['caption_pn'][i][k] = pn2id.get(pn_symbol, unknown)
                batch['caption_pn_length'][i] = len(pn)
                rpn = caption.polish_notation(reverse=True)
                assert len(rpn) <= pn_size, (len(rpn), pn_size, rpn)
                for k, pn_symbol in enumerate(rpn):
                    assert pn_symbol in pn2id, (pn_symbol, pn2id)
                    batch['caption_rpn'][i][k] = pn2id.get(pn_symbol, unknown)
                batch['caption_rpn_length'][i] = len(rpn)
                if include_model:
                    batch['caption_model'][i] = caption.model()

            if alternatives and self.worlds_per_instance > 1:
                from shapeworld.captions import PragmaticalPredication
                batch['alternatives'][i] = self.worlds_per_instance
                batch['world'][i].extend(batch['world'][i][0].copy() for _ in range(self.worlds_per_instance - 1))
                batch['world'][i][0] = self.apply_pixel_noise(world=world.get_array(world_array=batch['world'][i][0]))
                if include_model:
                    batch['world_model'][i].append(world.model())

                for j in range(1, self.worlds_per_instance):
                    correct = random() < correct_ratio

                    while True:
                        world = self.world_generator()
                        if world is None:
                            continue

                        caption = self.world_captioner.get_correct_caption()
                        predication = PragmaticalPredication(agreeing=world.entities)
                        caption.apply_to_predication(predication=predication)
                        agreement = caption.agreement(predication=predication, world=world)
                        if not correct:
                            if agreement >= 0.0:
                                continue
                            predication = PragmaticalPredication(agreeing=world.entities)
                            if not self.world_captioner.incorrect(caption=caption, predication=predication, world=world):
                                continue
                            agreement = caption.agreement(predication=predication, world=world)
                        if agreement > 0.0:
                            break

                    batch['world'][i][j] = self.apply_pixel_noise(world=world.get_array(world_array=batch['world'][i][j]))
                    if include_model:
                        batch['world_model'][i].append(world.model())
                    batch['agreement'][i].append(float(correct))

            else:
                batch['world'][i] = self.apply_pixel_noise(world=world.get_array(world_array=batch['world'][i]))
                if include_model:
                    batch['world_model'][i] = world.model()

        word2id = self.vocabularies['language']
        unknown = word2id['[UNKNOWN]']
        caption_size = self.vector_shape('caption')[0]

        unused_words = set(word2id)  # for assert
        unused_words.remove('')
        unused_words.remove('[UNKNOWN]')
        missing_words = set()  # for assert
        max_caption_size = caption_size  # for assert

        assert len(captions) == n * self.captions_per_instance if alternatives else len(captions) == n
        captions = self.caption_realizer.realize(captions=captions)

        for i, caption in enumerate(captions):
            caption = util.sentence2tokens(sentence=caption)

            if len(caption) > caption_size:
                if len(caption) > max_caption_size:
                    max_caption_size = len(caption)
                continue

            if alternatives and self.captions_per_instance > 1:
                j = i % self.captions_per_instance
                i = i // self.captions_per_instance
                batch['caption_length'][i].append(len(caption))
                caption_array = batch['caption'][i][j]
            else:
                batch['caption_length'][i] = len(caption)
                caption_array = batch['caption'][i]

            for k, word in enumerate(caption):
                if word in word2id:
                    unused_words.discard(word)
                else:
                    missing_words.add(word)
                caption_array[k] = word2id.get(word, unknown)

        if util.debug() and len(unused_words) > 0:
            print('Words unused in vocabulary: \'{}\''.format('\', \''.join(sorted(unused_words))))
        if util.debug() and max_caption_size < caption_size:
            print('Caption size smaller than max size: {} < {}'.format(max_caption_size, caption_size))
        if len(missing_words) > 0:
            print('Words missing in vocabulary: \'{}\''.format('\', \''.join(sorted(missing_words))))
        if max_caption_size > caption_size:
            print('Caption size exceeds max size: {} > {}'.format(max_caption_size, caption_size))
        assert not missing_words, missing_words
        assert max_caption_size <= caption_size, (max_caption_size, caption_size)

        return batch

    def get_html(self, generated, image_format='bmp', image_dir=''):
        id2word = self.vocabulary(value_type='language')
        worlds = generated['world']
        captions = generated['caption']
        caption_lengths = generated['caption_length']
        agreements = generated['agreement']

        data_html = list()
        for n, (world, caption, caption_length, agreement) in enumerate(zip(worlds, captions, caption_lengths, agreements)):

            if self.worlds_per_instance > 1 or self.captions_per_instance > 1:
                data_html.append('<div class="instance">')
            else:
                if agreement == 1.0:
                    agreement = 'correct'
                elif agreement == 0.0:
                    agreement = 'incorrect'
                else:
                    agreement = 'ambiguous'
                data_html.append('<div class="{agreement}">'.format(agreement=agreement))

            if self.worlds_per_instance > 1:
                for i, agreement in enumerate(agreement):
                    if agreement == 1.0:
                        agreement = 'correct'
                    elif agreement == 0.0:
                        agreement = 'incorrect'
                    else:
                        agreement = 'ambiguous'
                    data_html.append('<div class="{agreement}" style="padding: 5px;"><div class="world"><img src="{image_dir}world-{world}-{alt}.{format}" alt="world-{world}-{alt}.{format}"></div></div>'.format(
                        agreement=agreement,
                        image_dir=image_dir,
                        world=n,
                        format=image_format,
                        alt=i
                    ))
            else:
                data_html.append('<div class="world"><img src="{image_dir}world-{world}.{format}" alt="world-{world}.{format}"></div>'.format(image_dir=image_dir, world=n, format=image_format))

            data_html.append('<div class="num"><b>({num})</b></div>'.format(num=(n + 1)))

            if self.captions_per_instance > 1:
                data_html.append('<div class="caption">')
                for caption, caption_length, agreement in zip(caption, caption_length, agreement):
                    if agreement == 1.0:
                        agreement = 'correct'
                    elif agreement == 0.0:
                        agreement = 'incorrect'
                    else:
                        agreement = 'ambiguous'
                    data_html.append('<div class="{agreement}">{caption}</div>'.format(
                        agreement=agreement,
                        caption=util.tokens2sentence(id2word[word] for word in caption[:caption_length])
                    ))
                data_html.append('</div>')
            else:
                data_html.append('<div class="caption">{caption}</div>'.format(
                    caption=util.tokens2sentence(id2word[word] for word in caption[:caption_length])
                ))

            data_html.append('</div>')

        html = '<!DOCTYPE html><html><head><title>{dtype} {name}</title><style>.data{{width: 100%; height: 100%;}} .instance{{width: 100%; display: flex; margin-top: 1px; margin-bottom: 1px; background-color: #DDEEFF; vertical-align: middle; align-items: center;}} .world{{height: {world_height}px; display: inline-block; flex-grow: 0; vertical-align: middle;}} .num{{width: 50px; display: inline-block; flex-grow: 0; text-align: center; vertical-align: middle; margin-left: 10px;}} .caption{{display: inline-block; flex-grow: 1; vertical-align: middle; margin-left: 10px;}} .correct{{margin-top: 1px; margin-bottom: 1px; background-color: #BBFFBB;}} .incorrect{{margin-top: 1px; margin-bottom: 1px; background-color: #FFBBBB;}} .ambiguous{{margin-top: 1px; margin-bottom: 1px; background-color: #FFFFBB;}}</style></head><body><div class="data">{data}</div></body></html>'.format(
            dtype=self.type,
            name=self.name,
            world_height=self.world_shape()[0],
            data=''.join(data_html)
        )
        return html
