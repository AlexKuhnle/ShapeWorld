import json
import os
from PIL import Image
from shapeworld import util
from shapeworld.world import World


def clevr(directory, mode, parts=None):
    worlds = images_iter(directory=directory, mode=mode, parts=parts)
    world_models = scenes_iter(directory=directory, mode=mode, parts=parts)
    questions_answers = questions_iter(directory=directory, mode=mode, parts=parts)
    world = next(worlds)
    world_model = next(world_models)
    questions = list()
    question_models = list()
    answers = list()
    current_index = 0
    while True:
        try:
            image_index, question, question_model, answer = next(questions_answers)
        except StopIteration:
            yield world, world_model, questions, question_models, answers
            break
        if image_index != current_index:
            yield world, world_model, questions, question_models, answers
            current_index += 1
            world = next(worlds)
            world_model = next(world_models)
            questions = list()
            question_models = list()
            answers = list()
        questions.append(question)
        question_models.append(question_model)
        answers.append(answer)
    try:
        next(worlds)
        assert False
    except StopIteration:
        pass
    try:
        next(world_models)
        assert False
    except StopIteration:
        pass


def images_iter(directory, mode, parts=None):
    split = 'val' if mode == 'validation' else mode
    if parts is not None:
        split += parts[mode]
    directory = os.path.join(directory, 'images', split)
    for root, dirs, files in os.walk(directory):
        assert root == directory
        assert not dirs
        for n in range(len(files)):
            filename = 'CLEVR_{}_{:0>6}.png'.format(split, n)
            files.remove(filename)
            with open(os.path.join(root, filename), 'rb') as filehandle:
                image = Image.open(fp=filehandle)
                world = World.from_image(image)
            yield world


def scenes_iter(directory, mode, parts=None):
    split = 'val' if mode == 'validation' else mode
    if parts is not None:
        split += parts[mode]
    path = os.path.join(directory, 'scenes', 'CLEVR_{}_scenes.json'.format(split))
    if os.path.isfile(path):
        with open(path, 'r') as filehandle:
            chars = filehandle.read(2)
            assert chars == '{"'
            chars = filehandle.read(1)
            while chars != 's':
                while filehandle.read(1) != '"':
                    pass
                chars = filehandle.read(3)
                assert chars == ': {'
                while filehandle.read(1) != '}':
                    pass
                chars = filehandle.read(3)
                assert chars == ', "'
                chars = filehandle.read(1)
            chars = filehandle.read(8)
            assert chars == 'cenes": '
            for n, scene_dict in enumerate(json_list_generator(fp=filehandle)):
                directions = scene_dict['directions']
                objects = scene_dict['objects']
                relationships = scene_dict['relationships']
                world_model = dict(directions=directions, objects=objects, relationships=relationships)
                assert scene_dict['image_index'] == n
                assert scene_dict['split'] == split
                assert scene_dict['image_filename'] == 'CLEVR_{}_{:0>6}.png'.format(split, n)
                assert len(scene_dict) == 6
                yield world_model
            chars = filehandle.read(1)
            while chars == ',':
                chars == filehandle.read(2)
                assert chars == ' "'
                while filehandle.read(1) != '"':
                    pass
                chars = filehandle.read(3)
                assert chars == ': {'
                while filehandle.read(1) != '}':
                    pass
                chars = filehandle.read(1)
            assert chars == '}'
            chars = filehandle.read()
            assert not chars
    else:
        directory = os.path.join(directory, 'images', split)
        for root, dirs, files in os.walk(directory):
            for n in range(len(files)):
                yield dict()


numbers = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven', '12': 'twelve'}


def questions_iter(directory, mode, parts=None):
    split = 'val' if mode == 'validation' else mode
    if parts is not None:
        split += parts[mode]
    path = os.path.join(directory, 'questions', 'CLEVR_{}_questions.json'.format(split))
    with open(path, 'r') as filehandle:
        chars = filehandle.read(2)
        assert chars == '{"'
        chars = filehandle.read(1)
        while chars != 'q':
            while filehandle.read(1) != '"':
                pass
            chars = filehandle.read(3)
            assert chars == ': {'
            while filehandle.read(1) != '}':
                pass
            chars = filehandle.read(3)
            assert chars == ', "'
            chars = filehandle.read(1)
        chars = filehandle.read(11)
        assert chars == 'uestions": '
        image_index = 0
        for n, question_dict in enumerate(json_list_generator(fp=filehandle)):
            if image_index != question_dict['image_index']:
                image_index += 1
                assert image_index == question_dict['image_index']
            question = question_dict['question'].lower()
            if question[-1] != '?':
                question += '?'
            question = util.string2tokens(string=question)
            if mode == 'test':
                question_model = dict()
                answer = '[UNKNOWN]'
            else:
                family = question_dict['question_family_index']
                program = question_dict['program']
                question_model = dict(family=family, program=program)
                answer = question_dict['answer'].lower()
            if answer in numbers:
                answer = numbers[answer]
            answer = util.string2tokens(string=answer)
            assert question_dict['question_index'] == n
            assert question_dict['split'] == split
            assert question_dict['image_filename'] == 'CLEVR_{}_{:0>6}.png'.format(split, image_index)
            assert len(question_dict) == 8 or (mode == 'test' and len(question_dict) == 5)
            yield image_index, question, question_model, answer
        chars = filehandle.read(1)
        while chars == ',':
            chars = filehandle.read(2)
            assert chars == ' "'
            while filehandle.read(1) != '"':
                pass
            chars = filehandle.read(3)
            assert chars == ': {'
            while filehandle.read(1) != '}':
                pass
            chars = filehandle.read(1)
        assert chars == '}'
        chars = filehandle.read()
        assert not chars


def json_list_generator(fp):
    char = fp.read(1)
    assert char == '['
    char = fp.read(1)
    assert char in '{['
    chars = [char]
    bracket = 1
    string = False
    while True:
        char = fp.read(1)
        if not char:
            assert False
        if char == '"':
            n = 1
            while chars[-n] == '\\':
                n += 1
            if n % 2 == 1:
                string = not string
        if string:
            chars.append(char)
            continue
        if char in '{[':
            bracket += 1
        if bracket == 0:
            if char == ']':
                return
            assert char in ', '
            continue
        chars.append(char)
        if char in '}]':
            bracket -= 1
            if bracket == 0:
                json_str = ''.join(chars)
                chars = []
                try:
                    yield json.loads(s=json_str)
                except Exception:
                    print('Could not read JSON.')
                    print('|', json_str, '|')
