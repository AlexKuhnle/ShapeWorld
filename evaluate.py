import argparse
from datetime import datetime
from importlib import import_module
import json
import os
import sys
from shapeworld import Dataset, util
from models.TFMacros.tf_macros import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('-t', '--type', help='Dataset type')
    parser.add_argument('-n', '--name', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Dataset name')
    parser.add_argument('-v', '--variant', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Label of configuration variant')
    parser.add_argument('-l', '--language', default=None, help='Language')
    parser.add_argument('-c', '--config', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Configuration file/directory')

    parser.add_argument('-m', '--model', help='Model')
    parser.add_argument('-y', '--hyperparams-file', default=None, help='Model hyperparameters file (default: hyperparams directory)')

    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=64, help='Batch size')
    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=100, help='Number of iterations')
    parser.add_argument('-q', '--query', default=None, help='Additional values to query (separated by commas)')
    parser.add_argument('-s', '--serialize', default=None, help='Values to serialize (separated by commas)')

    parser.add_argument('--model-dir', help='TensorFlow model directory, storing the model computation graph and parameters')
    parser.add_argument('--report-file', default=None, help='CSV file reporting the evaluation results')

    parser.add_argument('--verbosity', type=int, choices=(0, 1, 2), default=1, help='Verbosity (0: no messages, 1: default, 2: plus TensorFlow messages)')

    parser.add_argument('--config-values', nargs=argparse.REMAINDER, default=(), help='Additional dataset configuration values passed as command line arguments')

    args = parser.parse_args()
    args.config_values = util.parse_config(values=args.config_values)

    # tensorflow verbosity
    if args.verbosity >= 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # dataset
    dataset = Dataset.create(dtype=args.type, name=args.name, variant=args.variant, language=args.language, config=args.config, **args.config_values)

    # information about dataset and model
    if args.verbosity >= 1:
        sys.stdout.write('{time} {dataset}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dataset=dataset))
        if args.config is None:
            if args.config_values:
                sys.stdout.write('         config: {config}\n'.format(config=args.config_values))
        else:
            sys.stdout.write('         config: {config}\n'.format(config=args.config))
            if args.config_values:
                sys.stdout.write('                 {config}\n'.format(config=args.config_values))
        sys.stdout.write('         {} model: {}\n'.format(args.type, args.model))
        sys.stdout.write('         hyperparameters: {}\n'.format(args.hyperparams_file))
        sys.stdout.flush()

    if args.type == 'agreement':
        dataset_parameters = dict(
            world_shape=dataset.world_shape(),
            vocabulary_size=dataset.vocabulary_size(value_type='language'),
            rpn_vocabulary_size=dataset.vocabulary_size(value_type='rpn')
        )
        for value_name in dataset.vectors:
            dataset_parameters[value_name + '_shape'] = dataset.vector_shape(value_name=value_name)
        query = ('agreement_accuracy',)
        serialize = ()

    elif args.type == 'classification':
        dataset_parameters = dict(
            world_shape=dataset.world_shape(),
            num_classes=dataset.num_classes,
            multi_class=dataset.multi_class,
            count_class=dataset.count_class
        )
        for value_name in dataset.vectors:
            dataset_parameters[value_name + '_shape'] = dataset.vector_shape(value_name=value_name)
        query = ('classification_fscore', 'classification_precision', 'classification_recall')
        serialize = ()

    elif args.type == 'clevr_classification':
        dataset_parameters = dict(
            world_shape=dataset.world_shape(),
            vocabulary_size=dataset.vocabulary_size,
            num_answers=len(dataset.answers)
        )
        for value_name in dataset.vectors:
            dataset_parameters[value_name + '_shape'] = dataset.vector_shape(value_name=value_name)
        query = ('answer_fscore', 'answer_precision', 'answer_recall')
        serialize = ()

    else:
        assert False

    if args.query:
        query = tuple(args.query.split(','))
    if args.serialize:
        serialize = tuple(args.serialize.split(','))
        query += serialize

    if args.hyperparams_file is None:
        hyperparams_file = os.path.join('models', dataset.type, 'hyperparams', args.model + '.params.json')
        if os.path.isfile(hyperparams_file):
            with open(hyperparams_file, 'r') as filehandle:
                parameters = json.load(fp=filehandle)
        else:
            parameters = dict()
    else:
        with open(args.hyperparams_file, 'r') as filehandle:
            parameters = json.load(fp=filehandle)

    # restore
    iteration_start = 1
    if args.report_file:
        with open(args.report_file, 'r') as filehandle:
            for line in filehandle:
                value = line.split(',')[0]
        if value != 'iteration':
            iteration_start = int(value) + 1

    with Model(name=args.model, learning_rate=parameters.pop('learning_rate', 1e-3), weight_decay=parameters.pop('weight_decay', None), clip_gradients=parameters.pop('clip_gradients', None), model_directory=args.model_dir) as model:
        parameters.pop('dropout_rate', None)

        module = import_module('models.{}.{}'.format(args.type, args.model))
        module.model(model=model, inputs=dict(), dataset_parameters=dataset_parameters, **parameters)  # no input tensors, hence None for placeholder creation
        model.finalize(restore=(args.model_dir is not None))

        if args.verbosity >= 1:
            sys.stdout.write('         parameters: {:,}\n'.format(model.num_parameters))
            sys.stdout.write('         bytes: {:,}\n'.format(model.num_bytes))
            sys.stdout.flush()

        if args.verbosity >= 1:
            sys.stdout.write('{} evaluate model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()

        train = {name: 0.0 for name in query}
        for _ in range(args.iterations):
            generated = dataset.generate(n=args.batch_size, mode='train')
            queried = model(query=query, data=generated)
            train = {name: value + queried[name] for name, value in train.items()}
        train = {name: value / args.iterations for name, value in train.items()}
        sys.stdout.write('         train: ')
        for name in query:
            sys.stdout.write('{}={:.3f} '.format(name, train[name]))
        sys.stdout.write('\n')
        sys.stdout.flush()
        if serialize:
            dataset.serialize(path=None, generated=generated, additional={name: (train[name], serialize[name]) for name in serialize})

        validation = {name: 0.0 for name in query}
        for _ in range(args.iterations):
            generated = dataset.generate(n=args.batch_size, mode='validation')
            queried = model(query=query, data=generated)
            validation = {name: value + queried[name] for name, value in validation.items()}
        validation = {name: value / args.iterations for name, value in validation.items()}
        sys.stdout.write('         validation: ')
        for name in query:
            sys.stdout.write('{}={:.3f} '.format(name, validation[name]))
        sys.stdout.write('\n')
        sys.stdout.flush()
        if serialize:
            dataset.serialize(path=None, generated=generated, additional={name: (validation[name], serialize[name]) for name in serialize})

        test = {name: 0.0 for name in query}
        for _ in range(args.iterations):
            generated = dataset.generate(n=args.batch_size, mode='test')
            queried = model(query=query, data=generated)
            test = {name: value + queried[name] for name, value in test.items()}
        test = {name: value / args.iterations for name, value in test.items()}
        sys.stdout.write('         test: ')
        for name in query:
            sys.stdout.write('{}={:.3f} '.format(name, test[name]))
        sys.stdout.write('\n')
        sys.stdout.flush()
        if serialize:
            dataset.serialize(path=None, generated=generated, additional={name: (test[name], serialize[name]) for name in serialize})

        if args.verbosity >= 1:
            sys.stdout.write('\n{} model evaluation finished\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
