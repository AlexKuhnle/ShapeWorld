import json
import os
from PIL import Image
from shapeworld import util
from shapeworld.world import World


def clevr(directory, mode, parts=None):
    if parts is not None:
        assert len(parts) == 3
        parts = dict(train=parts[0], validation=parts[1], test=parts[2])
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
            question = util.sentence2tokens(sentence=question)
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
            answer = util.sentence2tokens(sentence=answer)
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


# CLEVR program vocab
# <NULL>, <START>, <END>, <UNK>
# scene, unique, exist, count,
# intersect, union,
# greater_than, less_than, equal_integer,
# equal_color, equal_material, equal_shape, equal_size,
# filter_size[large], filter_size[small],
# "query_color": 29, "query_material": 30, "query_shape": 31, "query_size": 32,
# "same_color": 37, "same_material": 38, "same_shape": 39, "same_size": 40


def parse_program(mode, model, index=0, inputs=None):
    assert 0 <= mode <= 1

    if model['component'] == 'Attribute':
        # predtype == 'relation' missing

        if mode == 0:
            assert len(inputs) == 1
            return [
                dict(inputs=inputs, function='attribute', value_inputs=['{}-{}'.format(model['predtype'], model['value'])])
            ]

        elif mode == 1:
            assert len(inputs) == 1
            if model['predtype'] in ('shape', 'color'):
                function = 'filter_' + model['predtype']
            elif model['predtype'] == 'texture':
                function = 'filter_material'
            else:
                assert False, model
            return [
                # dict(inputs=[], function='scene', value_inputs=[]),
                dict(inputs=inputs, function=function, value_inputs=['{}-{}'.format(model['predtype'], model['value'])])
            ]

        else:
            assert False, mode

    elif model['component'] == 'EntityType':

        if mode == 0 or mode == 1:
            modules = list()
            if inputs is None:
                modules.append(dict(inputs=[], function='scene', value_inputs=[]))
                inputs = [index]
            else:
                assert len(inputs) == 1
            if len(model['value']) > 0:
                modules += parse_program(mode=mode, model=model['value'][0], index=(index + len(modules)), inputs=inputs)
                for model in model['value'][1:]:
                    modules += parse_program(mode=mode, model=model, index=(index + len(modules)), inputs=[index + len(modules) - 1])
            return modules

        else:
            assert False, mode

    elif model['component'] == 'Relation':
        # should connect relation???

        if mode == 0:
            assert len(inputs) == 1
            if model['predtype'] in ('attribute', 'type'):
                modules = parse_program(mode=mode, model=model['value'], index=index, inputs=inputs)
                return modules + [
                    dict(inputs=[index + len(modules) - 1], function='relation', value_inputs=[model['predtype']])
                ]
            else:
                reference = parse_program(mode=mode, model=model['reference'], index=index)
                if 'comparison' in model:
                    comparison = parse_program(mode=mode, model=model['comparison'], index=(index + len(reference)))
                    return reference + comparison + [
                        dict(inputs=[inputs[0], index + len(reference) - 1, index + len(reference) + len(comparison) - 1], function='relation', value_inputs=['{}({})'.format(model['predtype'], model['value'])])
                    ]
                else:
                    return reference + [
                        dict(inputs=[inputs[0], index + len(reference) - 1], function='relation', value_inputs=['{}-{}'.format(model['predtype'], model['value'])])
                    ]

        elif mode == 1:
            assert inputs is None
            if model['predtype'] in ('attribute', 'type'):
                modules = [dict(inputs=[], function='scene', value_inputs=[])]
                modules.extend(parse_program(mode=mode, model=model['value'], index=index, inputs=[index]))
                return modules + [
                    dict(inputs=[index + len(modules) - 1], function='relate', value_inputs=[model['predtype']])
                ]
            else:
                assert False, model

        else:
            assert False, mode

    elif model['component'] == 'Existential':
        # should connect relation???

        if mode == 0:
            assert inputs is None
            restrictor = parse_program(mode=mode, model=model['restrictor'], index=index)
            body = parse_program(mode=mode, model=model['body'], index=(index + len(restrictor)), inputs=[index + len(restrictor) - 1])
            return restrictor + body + [
                dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(body) - 1], function='quantifier', value_inputs=['existential'])
            ]

        elif mode == 1:
            assert inputs is None
            restrictor = parse_program(mode=mode, model=model['restrictor'], index=index)
            body = parse_program(mode=mode, model=model['body'], index=(index + len(restrictor)))
            return restrictor + body + [
                dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(body) - 1], function='intersect', value_inputs=[]),
                dict(inputs=[index + len(restrictor) + len(body)], function='exist', value_inputs=[])
            ]

        else:
            assert False, mode

    elif model['component'] == 'Quantifier':

        if mode == 0:
            assert inputs is None
            restrictor = parse_program(mode=mode, model=model['restrictor'], index=index)
            body = parse_program(mode=mode, model=model['body'], index=(index + len(restrictor)), inputs=[index + len(restrictor) - 1])
            return restrictor + body + [
                dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(body) - 1], function='quantifier', value_inputs=['{}-{}-{}'.format(model['qtype'], model['qrange'], model['quantity'])])
            ]

        elif mode == 1:
            assert inputs is None
            restrictor = parse_program(mode=mode, model=model['restrictor'], index=index)
            body = parse_program(mode=mode, model=model['body'], index=(index + len(restrictor)))
            return restrictor + body + [
                dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(body) - 1], function='intersect', value_inputs=[]),
                dict(inputs=[index + len(restrictor) - 1], function='count', value_inputs=[]),
                dict(inputs=[index + len(restrictor) + len(body)], function='count', value_inputs=[]),
                # equal_integer can take argument? tbd
                dict(inputs=[index + len(restrictor) + len(body) + 1, index + len(restrictor) + len(body) + 2], function='equal_integer', value_inputs=['{}-{}-{}'.format(model['qtype'], model['qrange'], model['quantity'])])
            ]

        else:
            assert False, mode

    elif model['component'] == 'NumberBound':

        if mode == 0:
            assert inputs is None
            quantifier = parse_program(mode=mode, model=model['quantifier'], index=index)
            return quantifier + [
                dict(inputs=[index + len(quantifier) - 1], function='number-bound', value_inputs=[str(model['bound'])])
            ]

        elif mode == 1:
            assert inputs is None
            assert False, model

        else:
            assert False, mode

    elif model['component'] == 'ComparativeQuantifier':

        if mode == 0:
            assert inputs is None
            restrictor = parse_program(mode=mode, model=model['restrictor'], index=index)
            comparison = parse_program(mode=mode, model=model['comparison'], index=(index + len(restrictor)))
            restrictor_body = parse_program(mode=mode, model=model['body'], index=(index + len(restrictor) + len(comparison)), inputs=[index + len(restrictor) - 1])
            comparison_body = parse_program(mode=mode, model=model['body'], index=(index + len(restrictor) + len(comparison)), inputs=[index + len(restrictor) + len(comparison) - 1])
            return restrictor + comparison + restrictor_body + comparison_body + [
                dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(comparison) - 1, index + len(restrictor) + len(comparison) + len(restrictor_body) - 1, index + len(restrictor) + len(comparison) + len(restrictor_body) + len(comparison_body) - 1], function='comparative-quantifier', value_inputs=['{}-{}-{}'.format(model['qtype'], model['qrange'], model['quantity'])])
            ]

        elif mode == 1:
            assert inputs is None
            assert False, model

        else:
            assert False, mode

    elif model['component'] == 'Proposition':
        assert inputs is None
        assert False, model

    else:
        assert False, model
