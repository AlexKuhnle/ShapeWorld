from collections import Counter
from datetime import datetime
from itertools import chain, combinations
import json
import os
import tarfile
import zipfile


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


def parse_config(string):
    assert string
    if string in ('none', 'None', 'null', 'Null'):
        return None
    elif string[0] == '{':
        assert string[-1] == '}'
        if '\'' in string and '\"' not in string:
            string = string.replace('\'', '\"')
        if '=' in string and ':' not in string:
            string = string.replace('=', ':')
        return json.loads(string)
    else:
        return string


def powerset(values, min_num=None, max_num=None):
    values = list(values)
    min_num = min_num or 0
    max_num = max_num or len(values)
    return chain.from_iterable(combinations(values, num) for num in range(min_num, max_num + 1))


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


def cumulative_distribution(values):
    if isinstance(values, int):
        assert values > 0
        return [n / values for n in range(1, values + 1)]
    elif all(isinstance(x, float) or isinstance(x, int) for x in values):
        assert all(x >= 0.0 for x in values)
        denominator = sum(values)
        prob = 0.0
        cdf = []
        for x in values:
            prob += x
            cdf.append(prob / denominator)
        return cdf
    else:
        assert False


class Archive(object):

    def __init__(self, directory, mode, name=None, archive=None):
        assert mode == 'r' or mode == 'w'
        assert archive in (None, 'zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma')
        self.archive = os.path.join(directory, name) if name else directory
        self.mode = mode
        self.temp_directory = str(datetime.now().timestamp())
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
