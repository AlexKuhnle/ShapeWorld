import argparse
from datetime import datetime
from importlib import import_module
import json
import os
import sys
from shapeworld import dataset, util
from models.TFMacros.tf_macros import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('-t', '--type', help='Dataset type')
    parser.add_argument('-n', '--name', help='Dataset name')
    parser.add_argument('-l', '--language', default=None, help='Dataset language')
    parser.add_argument('-c', '--config', type=util.parse_config, default=None, help='Dataset configuration file')
    parser.add_argument('-p', '--pixel-noise', type=float, default=0.1, help='Pixel noise range')

    parser.add_argument('-m', '--model', help='Model')
    parser.add_argument('-y', '--hyperparams-file', default=None, help='Model hyperparameters file (default: hyperparams directory)')

    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=1, help='Iterations')
    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=1000, help='Batch size')

    parser.add_argument('--model-dir', help='TensorFlow model directory, storing the model computation graph and parameters')
    parser.add_argument('--report-file', default=None, help='CSV file reporting the evaluation results')

    parser.add_argument('-v', '--verbosity', type=int, choices=(0, 1, 2), default=1, help='Verbosity (0: nothing, 1: default, 2: TensorFlow)')

    parser.add_argument('--query', default=None, help='Experimental: Values to query (separated by commas)')
    parser.add_argument('--serialize', default=None, help='Experimental: Values to serialize (requires --evaluate) (separated by commas)')
    args = parser.parse_args()

    # import tensorflow
    if args.verbosity >= 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    assert args.model_dir is not None

    # dataset
    dataset = dataset(dtype=args.type, name=args.name, language=args.language, config=args.config)

    # information about dataset and model
    if args.verbosity >= 1:
        sys.stdout.write('{time} {dataset}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dataset=dataset))
        sys.stdout.write('         config: {}\n'.format(args.config))
        sys.stdout.write('         {} model: {}\n'.format(args.type, args.model))
        sys.stdout.write('         hyperparameters: {}\n'.format(args.hyperparameters))
        sys.stdout.flush()

    if args.type == 'agreement':
        parameters = dict(
            world_shape=dataset.world_shape,
            vocabulary_size=dataset.vocabulary_size(value_type='language'),
            caption_shape=dataset.vector_shape(value_name='caption')
        )
        query = ('agreement_accuracy',)
        serialize = ()

    elif args.type == 'classification':
        parameters = dict(
            world_shape=dataset.world_shape,
            num_classes=dataset.num_classes,
            multi_class=dataset.multi_class,
            class_count=dataset.class_count
        )
        query = ('classification_fscore', 'classification_precision', 'classification_recall')
        serialize = ()

    elif args.type == 'clevr_classification':
        parameters = dict(
            world_shape=dataset.world_shape,
            vocabulary_size=dataset.vocabulary_size,
            question_shape=dataset.vector_shape('question'),
            num_answers=len(dataset.answers)
        )
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
        with open(os.path.join('models', dataset.type, 'hyperparams', args.model + '.params.json'), 'r') as filehandle:
            parameters.update(json.load(fp=filehandle))
    else:
        with open(args.hyperparams_file, 'r') as filehandle:
            parameters.update(json.load(fp=filehandle))

    # restore
    iteration_start = 1
    if args.report_file:
        with open(args.report_file, 'r') as filehandle:
            for line in filehandle:
                value = line.split(',')[0]
        if value != 'iteration':
            iteration_start = int(value) + 1

    with Model(name=args.model, learning_rate=parameters.pop('learning_rate'), weight_decay=parameters.pop('weight_decay', 0.0), model_directory=args.model_dir) as model:
        parameters.pop('dropout_rate', 0.0)

        module = import_module('models.{}.{}'.format(args.type, args.model))
        module.model(model=model, inputs=dict(), **parameters)  # no input tensors, hence None for placeholder creation
        model.finalize(restore=True)

        if args.verbosity >= 1:
            sys.stdout.write('         parameters: {:,}\n'.format(model.num_parameters))
            sys.stdout.write('         bytes: {:,}\n'.format(model.num_bytes))
            sys.stdout.flush()

        if args.verbosity >= 1:
            sys.stdout.write('{} evaluate model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()

        train = {name: 0.0 for name in query}
        for _ in range(args.iterations):
            generated = dataset.generate(n=args.batch_size, mode='train', noise_range=args.pixel_noise)
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
            generated = dataset.generate(n=args.batch_size, mode='validation', noise_range=args.pixel_noise)
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
            generated = dataset.generate(n=args.batch_size, mode='test', noise_range=args.pixel_noise)
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

        # else:  # training
        #     if args.verbosity >= 1:
        #         sys.stdout.write('{} train model...\n'.format(datetime.now().strftime('%H:%M:%S')))
        #         sys.stdout.flush()
        #     before = datetime.now()

        #     if args.tf_records:
        #         mean = {name: 0.0 for name in query}
        #         n = 0
        #         for iteration in range(iteration_start, iteration_end + 1):
        #             train = model(query=query, optimize=True, dropout=args.dropout_rate)  # loss !!!???
        #             mean = {name: value + train[name] for name, value in mean.items()}
        #             n += 1
        #             if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
        #                 mean = {name: value / n for name, value in mean.items()}
        #                 after = datetime.now()
        #                 if args.verbosity >= 1:
        #                     sys.stdout.write('\r         {:.0f}%  {}/{}  '.format(iteration * 100 / iteration_end, iteration, iteration_end))
        #                     for name in query:
        #                         sys.stdout.write('{}={:.3f}  '.format(name, train[name]))
        #                     sys.stdout.write('(time per evaluation iteration: {})'.format(str(after - before).split('.')[0]))
        #                     sys.stdout.flush()
        #                 before = datetime.now()
        #                 if args.report_file:
        #                     with open(args.report_file, 'a') as filehandle:
        #                         filehandle.write(str(iteration))
        #                         for name in query:
        #                             filehandle.write(',' + str(train[name]))
        #                         filehandle.write('\n')
        #                 mean = {name: 0.0 for name in mean}
        #                 n = 0

        #     else:
        #         for iteration in range(iteration_start, iteration_end + 1):
        #             generated = dataset.generate(n=args.batch_size, mode='train', noise_range=args.pixel_noise)
        #             model(data=generated, optimize=True, dropout=args.dropout_rate)
        #             if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
        #                 generated = dataset.generate(n=args.evaluation_size, mode='train', noise_range=args.pixel_noise)
        #                 train = model(query=query, data=generated)
        #                 generated = dataset.generate(n=args.evaluation_size, mode='validation', noise_range=args.pixel_noise)
        #                 validation = model(query=query, data=generated)
        #                 after = datetime.now()
        #                 if args.verbosity >= 1:
        #                     sys.stdout.write('\r         {:.0f}%  {}/{}  '.format(iteration * 100 / iteration_end, iteration, iteration_end))
        #                     sys.stdout.write('train: ')
        #                     for name in query:
        #                         sys.stdout.write('{}={:.3f} '.format(name, train[name]))
        #                     sys.stdout.write(' validation: ')
        #                     for name in query:
        #                         sys.stdout.write('{}={:.3f} '.format(name, validation[name]))
        #                     sys.stdout.write(' (time per evaluation iteration: {})'.format(str(after - before).split('.')[0]))
        #                     sys.stdout.flush()
        #                 before = datetime.now()
        #                 if args.report_file:
        #                     with open(args.report_file, 'a') as filehandle:
        #                         filehandle.write(str(iteration))
        #                         for name in query:
        #                             filehandle.write(',' + str(train[name]))
        #                         for name in query:
        #                             filehandle.write(',' + str(validation[name]))
        #                         filehandle.write('\n')
