from __future__ import division
from collections import Counter
from itertools import chain, combinations
import json
from math import sqrt
import os
from random import randint, random, randrange, uniform
import tarfile
import time
import zipfile


_debug = False


def debug():
    global _debug
    return _debug


def value_or_default(value, default):
    if value is None:
        return default
    else:
        return value


def mode_specific_lists(general, train, validation, test, allow_none=False):
    assert (general is None) or (train is None) or (test is None) or (set(general) == set(train) | set(test))
    if general is None:
        if test is None:
            assert allow_none
            general = None
            train = None
            validation = None
            test = None
        elif train is None:
            assert allow_none
            general = None
            train = None
            validation = None if validation is None else list(validation)
            test = list(test)
        else:
            train = list(train)
            validation = list(train) if validation is None else list(validation)
            test = list(test)
            general = list(set(train + validation + test))
    elif train is not None:
        general = list(general)
        train = list(train)
        validation = list(train) if validation is None else list(n for n in general if n not in train)
        test = list(n for n in general if n not in train)
    elif test is not None:
        general = list(general)
        train = list(n for n in general if n not in validation and n not in test)
        validation = list(train) if validation is None else list(validation)
        test = list(test)
    else:
        general = list(general)
        train = list(general)
        validation = list(general)
        test = list(general)
    return general, train, validation, test


def class_name(string):
    return ''.join(part[0].upper() + part[1:] for part in string.split('_'))


def real_name(string):
    return string[0].lower() + ''.join('_' + char.lower() if char.isupper() else char for char in string[1:])


def negative_response(response):
    return response.lower() in ('n', 'no', 'c', 'cancel', 'a', 'abort')


def parse_int_with_factor(string):
    assert string
    if len(string) < 2:
        return int(string)
    elif string[-1] == 'k':
        return int(string[:-1]) * 1000
    elif string[-1] == 'M':
        return int(string[:-1]) * 1000000
    elif string[-2:] == 'Ki':
        return int(string[:-2]) * 1024
    elif string[-2:] == 'Mi':
        return int(string[:-2]) * 1048576
    else:
        return int(string)


def parse_tuple(parse_item, unary_tuple=True, valid_sizes=None):
    def parse(string):
        if ',' in string:
            if string[0] == '(' and string[-1] == ')':
                assert len(string) > 2
                xs = string[1:-1].split(',')
                assert valid_sizes is None or len(xs) in valid_sizes
                return tuple(parse_item(x) for x in xs)
            else:
                xs = string.split(',')
                assert valid_sizes is None or len(xs) in valid_sizes
                return tuple(parse_item(x) for x in xs)
        elif unary_tuple:
            return (parse_item(string),)
        else:
            return parse_item(string)
    return parse


def parse_config(values):
    assert len(values) % 2 == 0
    config = dict()
    for key, value in zip(values[::2], values[1::2]):
        if key[0:2] == '--':
            key = key[2:].replace('-', '_')
        else:
            key = key.replace('-', '_')
        try:
            config[key] = json.loads(value)
        except json.decoder.JSONDecodeError:
            config[key] = value
    return config


def sentence2tokens(sentence):
    sentence = sentence[0].lower() + sentence[1:]
    return sentence.replace(', ', ' , ').replace('; ', ' ; ').replace('.', ' .').replace('?', ' ?').split()


def tokens2sentence(tokens):
    sentence = ' '.join(tokens).replace(' , ', ', ').replace(' ; ', '; ').replace(' .', '.').replace(' ?', '?')
    return sentence[0].upper() + sentence[1:]


def alternatives_type(value_type):
    if len(value_type) > 5 and value_type[:13] == 'alternatives(' and value_type[-1] == ')':
        return value_type[13:-1], True
    else:
        return value_type, False


def product(xs):
    prod = 1
    for x in xs:
        prod *= x
    return prod


def powerset(values, min_num=None, max_num=None):
    values = list(values)
    min_num = min_num or 0
    max_num = max_num or len(values)
    return chain.from_iterable(combinations(values, num) for num in range(min_num, max_num + 1))


def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    merged.update(dict2)
    return merged


def all_and_any(xs):
    try:
        if not next(xs):
            return False
    except StopIteration:
        return False
    return all(xs)


def any_or_none(xs):
    try:
        if next(xs):
            return True
    except StopIteration:
        return True
    return any(xs)


def any_not_all(xs):
    try:
        first = next(xs)
    except StopIteration:
        return False
    if first:
        return not all(xs)
    else:
        return any(xs)


# partial_order is dict: x -> set({>x})
def toposort(partial_order):
    result = []
    smaller = Counter(y for ys in partial_order.values() for y in ys)
    smallest = [x for x in partial_order if not smaller[x]]
    while smallest:
        x = smallest.pop()
        result.append(x)
        for y in partial_order[x]:
            smaller[y] -= 1
            if not smaller[y]:
                smallest.append(y)
    assert not list(smaller.elements())
    return result


def quadratic_uniform(a, b):
    return sqrt(uniform(a * a, b * b))


def cumulative_distribution(values):
    if isinstance(values, int):
        assert values > 0
        return [n / values for n in range(1, values + 1)]
    elif all(isinstance(x, float) or isinstance(x, int) for x in values):
        denominator = sum(values)
        prob = 0.0
        cdf = []
        for x in values:
            prob += max(x, 0.0)  # negative values are zero
            cdf.append(prob / denominator)
        return cdf
    else:
        assert False


def sample(cumulative_distribution, items=None):
    sample = random()
    if items:
        for item, prob in zip(items, cumulative_distribution):
            if sample < prob:
                return item
    else:
        for index, prob in enumerate(cumulative_distribution):
            if sample < prob:
                return index


# def sample_softmax(logits, temperature=1.0):
#     probabilities = [exp(logit / temperature) for logit in logits]
#     probabilities /= sum(probabilities)
#     return sample(cumulative_distribution=cumulative_distribution(values=probabilities))


def choice(items, num_range, auxiliary=None):
    items = list(items)
    if isinstance(num_range, int):
        num_items = num_range
    else:
        num_items = randint(*num_range)
    if len(items) == num_items:
        return items
    elif len(items) < num_items:
        chosen = items
        auxiliary = list(auxiliary)
        for _ in range(num_items - len(items)):
            pick = randrange(len(auxiliary))
            chosen.append(auxiliary.pop(pick))
        return chosen
    else:
        chosen = list()
        for _ in range(num_items):
            pick = randrange(len(items))
            chosen.append(items.pop(pick))
        return chosen


class Archive(object):

    def __init__(self, path, mode, archive=None):
        assert mode == 'r' or mode == 'w'
        assert archive in (None, 'zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma')
        self.archive = path
        self.mode = mode
        if not os.path.isdir(self.archive[:self.archive.rindex('/')]):
            os.mkdir(self.archive[:self.archive.rindex('/')])
        try:
            if not os.path.isdir('/tmp/shapeworld'):
                os.makedirs('/tmp/shapeworld')
            self.temp_directory = os.path.join('/tmp/shapeworld', 'temp-' + str(time.time()))
            os.mkdir(self.temp_directory)
        except PermissionError:
            self.temp_directory = os.path.join(self.archive[:self.archive.rindex('/')], 'temp-' + str(time.time()))
            os.mkdir(self.temp_directory)
        if archive is None:
            self.archive_type = None
            if not os.path.isdir(self.archive):
                os.mkdir(self.archive)
        elif archive[:3] == 'zip':
            self.archive_type = 'zip'
            if len(archive) == 3:
                compression = zipfile.ZIP_DEFLATED
            elif archive[4:] == 'none':
                compression = zipfile.ZIP_STORED
            elif archive[4:] == 'deflate':
                compression = zipfile.ZIP_DEFLATED
            elif archive[4:] == 'bzip2':
                compression = zipfile.ZIP_BZIP2
            elif archive[4:] == 'lzma':
                compression = zipfile.ZIP_LZMA
            if not self.archive.endswith('.zip'):
                self.archive += '.zip'
            self.archive = zipfile.ZipFile(self.archive, mode, compression)
        elif archive[:3] == 'tar':
            self.archive_type = 'tar'
            if len(archive) == 3:
                mode += ':gz'
                extension = '.gz'
            elif archive[4:] == 'none':
                extension = ''
            elif archive[4:] == 'gzip':
                mode += ':gz'
                extension = '.gz'
            elif archive[4:] == 'bzip2':
                mode += ':bz2'
                extension = '.bz2'
            elif archive[4:] == 'lzma':
                mode += ':xz'
                extension = '.lzma'
            if not self.archive.endswith('.tar' + extension):
                self.archive += '.tar' + extension
            self.archive = tarfile.open(self.archive, mode)

    def close(self):
        if self.archive_type is not None:
            self.archive.close()
        os.rmdir(self.temp_directory)

    def __enter__(self):
        if self.mode == 'r':
            return self.read_file
        else:
            return self.write_file

    def __exit__(self, type, value, traceback):
        self.close()
        return False

    def read_file(self, filename, binary=False):
        if self.archive_type is None:
            filename = os.path.join(self.archive, filename)
            if not os.path.isfile(filename):
                return None
            with open(filename, 'rb' if binary else 'r') as filehandle:
                value = filehandle.read()
            return value
        elif self.archive_type == 'zip':
            try:
                fileinfo = self.archive.getinfo(filename)
            except KeyError:
                return None
            value = self.archive.read(fileinfo)
            if not binary:
                value = value.decode()
            return value
        elif self.archive_type == 'tar':
            try:
                fileinfo = self.archive.getmember(filename)
            except KeyError:
                return None
            self.archive.extract(member=fileinfo, path=self.temp_directory)
            filepath = os.path.join(self.temp_directory, filename)
            with open(filepath, 'rb' if binary else 'r') as filehandle:
                value = filehandle.read()
            os.remove(filepath)
            return value

    def write_file(self, filename, value, binary=False):
        if self.archive_type is None:
            filename = os.path.join(self.archive, filename)
            with open(filename, 'wb' if binary else 'w') as filehandle:
                filehandle.write(value)
        elif self.archive_type == 'zip':
            if binary:
                filepath = os.path.join(self.temp_directory, filename)
                with open(filepath, 'wb') as filehandle:
                    filehandle.write(value)
                self.archive.write(filepath, filename)
                os.remove(filepath)
            else:
                self.archive.writestr(filename, value)
        elif self.archive_type == 'tar':
            filepath = os.path.join(self.temp_directory, filename)
            with open(filepath, 'wb' if binary else 'w') as filehandle:
                filehandle.write(value)
            self.archive.add(filepath, filename)
            os.remove(filepath)
