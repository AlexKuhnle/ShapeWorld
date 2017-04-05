import os
import sys


# dmrs sub-directory
directory = os.path.dirname(os.path.realpath(__file__))

# add pydmrs submodule to Python path
sys.path.insert(1, os.path.join(directory, 'pydmrs'))

# unpack grammar if used for the first time
if not os.path.isfile(os.path.join(directory, 'resources', 'erg-shapeworld.dat')):
    assert os.path.isfile(os.path.join(directory, 'resources', 'erg-shapeworld.dat.tar.gz'))
    import tarfile
    with tarfile.open(os.path.join(directory, 'resources', 'erg-shapeworld.dat.tar.gz'), 'r:gz') as filehandle:
        try:
            fileinfo = filehandle.getmember('erg-shapeworld.dat')
        except KeyError:
            assert False
        filehandle.extract(member=fileinfo)
