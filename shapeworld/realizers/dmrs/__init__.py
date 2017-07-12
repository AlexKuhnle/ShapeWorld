import os
import sys
from shapeworld.realizers.dmrs.realizer import DmrsRealizer


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


# check whether all caption components are registered properly
modifiers = set()
for key1, value1 in DmrsRealizer.modifiers.items():
    for key2 in value1:
        modifiers.add((key1, key2))
for key in DmrsRealizer.modifier_by_name.values():
    modifiers.remove(key)
assert not modifiers

relations = set()
for key1, value1 in DmrsRealizer.relations.items():
    for key2 in value1:
        relations.add((key1, key2))
for key in DmrsRealizer.relation_by_name.values():
    relations.remove(key)
assert not relations

quantifiers = set()
for key1, value1 in DmrsRealizer.quantifiers.items():
    for key2, value2 in value1.items():
        for key3 in value2:
            quantifiers.add((key1, key2, key3))
for key in DmrsRealizer.quantifier_by_name.values():
    quantifiers.remove(key)
assert not quantifiers
