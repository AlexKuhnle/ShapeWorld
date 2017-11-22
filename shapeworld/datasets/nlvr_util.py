import json
import os
import numpy as np
from PIL import Image
from shapeworld import util
from shapeworld.world import World


def nlvr(directory, mode):
    descriptions = dict()
    for identifier, world_models, description, label in descriptions_iter(directory=directory, mode=mode):
        assert identifier not in descriptions
        descriptions[identifier] = (world_models, description, label)
    counter = {identifier: 6 for identifier in descriptions}
    for identifier, worlds in images_iter(directory=directory, mode=mode):
        counter[identifier] -= 1
        world_models, description, agreement = descriptions[identifier]
        yield worlds, world_models, description, agreement
    assert all(count == 0 for count in counter.values())


def images_iter(directory, mode):
    mode = 'dev' if mode == 'validation' else mode
    directory = os.path.join(directory, mode, 'images')
    for root, dirs, files in os.walk(directory):
        if root == directory:
            assert not files
        else:
            assert not dirs
            for filename in files:
                assert filename[:len(mode) + 1] == mode + '-'
                identifier = filename[len(mode) + 1: -6]
                assert identifier[-2:] in ('-0', '-1', '-2', '-3')
                with open(os.path.join(root, filename), 'rb') as filehandle:
                    image = np.array(object=Image.open(fp=filehandle))
                    world1 = World.from_image(image[:, :100, :])
                    world2 = World.from_image(image[:, 150:250, :])
                    world3 = World.from_image(image[:, 300:, :])
                yield identifier, (world1, world2, world3)


def descriptions_iter(directory, mode):
    mode = 'dev' if mode == 'validation' else mode
    path = os.path.join(directory, mode, mode + '.json')
    with open(path, 'r') as filehandle:
        for line in filehandle:
            line = line.strip()
            description_dict = json.loads(s=line)
            identifier = description_dict['identifier']
            assert identifier[-2:] in ('-0', '-1', '-2', '-3')
            world_model1, world_model2, world_model3 = description_dict['structured_rep']
            description = description_dict['sentence'].lower()
            if description[-1] != '.':
                description += '.'
            description = util.string2tokens(string=description)
            agreement = description_dict['label']
            assert agreement in ('true', 'false')
            agreement = (agreement == 'true')
            assert len(description_dict['evals']) == (1 if mode == 'train' else 5)
            assert len(description_dict) == 5
            yield identifier, (world_model1, world_model2, world_model3), description, agreement
