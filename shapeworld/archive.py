import os
import tarfile
import zipfile


class Archive(object):

    def __init__(self, directory, mode, archive=None):
        assert mode == 'r' or mode == 'w'
        assert archive in (None, 'zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma')
        self.directory = directory
        self.mode = mode
        self.temp_path = os.path.join(directory, 'temp')
        if archive is None:
            self.archive_type = None
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
            self.archive = zipfile.ZipFile(os.path.join(directory, 'data.zip'), mode, compression)
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
            self.archive = tarfile.open(os.path.join(directory, 'data.tar' + extension), mode)

    def close(self):
        if self.archive_type is not None:
            self.archive.close()
        if os.path.isfile(self.temp_path):
            os.remove(self.temp_path)

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
            filename = os.path.join(self.directory, filename)
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
            self.archive.extract(member=fileinfo)
            os.rename(filename, self.temp_path)
            with open(self.temp_path, 'rb' if binary else 'r') as filehandle:
                value = filehandle.read()
            return value

    def write_file(self, filename, value, binary=False):
        if self.archive_type is None:
            filename = os.path.join(self.directory, filename)
            with open(filename, 'wb' if binary else 'w') as filehandle:
                filehandle.write(value)
        elif self.archive_type == 'zip':
            if binary:
                with open(self.temp_path, 'wb') as filehandle:
                    filehandle.write(value)
                self.archive.write(self.temp_path, filename)
            else:
                self.archive.writestr(filename, value)
        elif self.archive_type == 'tar':
            with open(self.temp_path, 'wb' if binary else 'w') as filehandle:
                filehandle.write(value)
            self.archive.add(self.temp_path, filename)
