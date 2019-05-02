import argparse
from datetime import datetime, timedelta
from importlib import import_module
import json
import os
import shutil
import sys
from shapeworld import Dataset, util
from models.TFMacros.tf_macros import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('-t', '--type', help='Dataset type')
    parser.add_argument('-n', '--name', type=util.parse_tuple(parse_item=str, unary_tuple=False), help='Dataset name')
    parser.add_argument('-v', '--variant', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Label of configuration variant')
    parser.add_argument('-l', '--language', default=None, help='Language')
    parser.add_argument('-c', '--config', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Configuration file/directory')

    parser.add_argument('-m', '--model', help='Model')
    parser.add_argument('-y', '--hyperparams-file', default=None, help='Model hyperparameters file (default: hyperparams directory)')
    parser.add_argument('-R', '--restore', action='store_true', help='Restore model (requires --model-file)')

    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=64, help='Batch size')
    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=1000, help='Number of iterations')
    parser.add_argument('-e', '--evaluation-iterations', type=util.parse_int_with_factor, default=10, help='Evaluation iterations')
    parser.add_argument('-f', '--evaluation-frequency', type=util.parse_int_with_factor, default=100, help='Evaluation frequency')
    parser.add_argument('-q', '--query', default=None, help='Additional values to query (separated by commas)')
    parser.add_argument('-T', '--tf-records', action='store_true', help='Use TensorFlow records')
    parser.add_argument('-F', '--features', action='store_true', help='Use image features (conv4 of resnet_v2_101) instead of raw image')

    parser.add_argument('--model-dir', default=None, help='TensorFlow model directory, storing the model computation graph and parameters')
    parser.add_argument('--save-frequency', type=int, default=3, help='Save frequency (in hours)')
    parser.add_argument('--summary-dir', default=None, help='TensorFlow summary directory for TensorBoard')
    parser.add_argument('--report-file', default=None, help='CSV file reporting the training results throughout the learning process')

    parser.add_argument('--verbosity', type=int, choices=(0, 1, 2), default=1, help='Verbosity (0: no messages, 1: default, 2: plus TensorFlow messages)')
    parser.add_argument('-Y', '--yes', action='store_true', help='Confirm all questions with yes')

    parser.add_argument('--config-values', nargs=argparse.REMAINDER, default=(), help='Additional dataset configuration values passed as command line arguments')

    args = parser.parse_args()
    args.config_values = util.parse_config(values=args.config_values)

    # TFRecords utility
    if args.tf_records:
        from shapeworld import tf_util

    # tensorflow verbosity
    if args.verbosity >= 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # dataset
    exclude_values = ('world',) if args.features else ('world_features',)
    dataset = Dataset.create(dtype=args.type, name=args.name, variant=args.variant, language=args.language, config=args.config, exclude_values=exclude_values, **args.config_values)

    # information about dataset and model
    if args.verbosity >= 1:
        sys.stdout.write('{time} train {model} on {dataset}\n'.format(
            time=datetime.now().strftime('%H:%M:%S'),
            model=args.model,
            dataset=dataset
        ))
        if args.config is None:
            if args.config_values:
                sys.stdout.write('         config: {config}\n'.format(config=args.config_values))
        else:
            sys.stdout.write('         config: {config}\n'.format(config=args.config))
            if args.config_values:
                sys.stdout.write('                 {config}\n'.format(config=args.config_values))
        sys.stdout.write('         hyperparameters: {}\n'.format(args.hyperparams_file))
        sys.stdout.flush()

    if dataset.type == 'agreement':
        dataset_parameters = dict(
            world_shape=dataset.world_shape(),
            vocabulary_size=dataset.vocabulary_size(value_type='language'),
            rpn_vocabulary_size=dataset.vocabulary_size(value_type='rpn')
        )
        for value_name in dataset.vectors:
            dataset_parameters[value_name + '_shape'] = dataset.vector_shape(value_name=value_name)
        query = ('agreement_accuracy',)

    elif dataset.type == 'classification':
        dataset_parameters = dict(
            world_shape=dataset.world_shape(),
            num_classes=dataset.num_classes,
            multi_class=dataset.multi_class,
            count_class=dataset.count_class
        )
        for value_name in dataset.vectors:
            dataset_parameters[value_name + '_shape'] = dataset.vector_shape(value_name=value_name)
        query = ('classification_fscore', 'classification_precision', 'classification_recall')

    elif dataset.type == 'clevr_classification':
        dataset_parameters = dict(
            world_shape=dataset.world_shape(),
            vocabulary_size=dataset.vocabulary_size(value_type='language'),
            num_answers=len(dataset.answers)
        )
        for value_name in dataset.vectors:
            dataset_parameters[value_name + '_shape'] = dataset.vector_shape(value_name=value_name)
        query = ('answer_fscore', 'answer_precision', 'answer_recall')

    else:
        assert False

    query += ('loss',)
    if args.query:
        query += tuple(args.query.split(','))

    if args.hyperparams_file is None:
        hyperparams_file = os.path.join('models', dataset.type, 'hyperparams', args.model + '.params.json')
        if os.path.isfile(hyperparams_file):
            with open(hyperparams_file, 'r') as filehandle:
                parameters = json.load(fp=filehandle)
        else:
            parameters = dict()
    elif os.path.isfile(args.hyperparams_file):
        with open(args.hyperparams_file, 'r') as filehandle:
            parameters = json.load(fp=filehandle)
    else:
        hyperparams_file = os.path.join('models', dataset.type, 'hyperparams', args.hyperparams_file + '.params.json')
        if os.path.isfile(hyperparams_file):
            with open(hyperparams_file, 'r') as filehandle:
                parameters = json.load(fp=filehandle)
        else:
            parameters = dict()

    # restore
    iteration_start = 1
    if args.restore:
        if args.report_file:
            lines = list()
            line_buffer = list()
            with open(args.report_file, 'r') as filehandle:
                for line in filehandle:
                    iteration, saved = line.split(',')[0:2]
                    if iteration == 'iteration':
                        lines.append(line)
                    else:
                        line_buffer.append(line)
                    if saved == 'yes':
                        iteration_start = int(iteration) + 1
                        lines.extend(line_buffer)
                        line_buffer = list()
            assert lines
            with open(args.report_file, 'w') as filehandle:
                filehandle.write(''.join(lines))

    else:
        if args.model_dir is not None:
            if os.path.isdir(args.model_dir):
                sys.stdout.write('Delete path {path}? '.format(path=args.model_dir))
                sys.stdout.flush()
                if args.yes:
                    sys.stdout.write('y\n')
                elif util.negative_response(sys.stdin.readline()[:-1]):
                    exit(0)
                shutil.rmtree(args.model_dir)
            os.makedirs(args.model_dir)
        if args.summary_dir is not None:
            if os.path.isdir(args.summary_dir):
                sys.stdout.write('Delete path {path}? '.format(path=args.summary_dir))
                sys.stdout.flush()
                if args.yes:
                    sys.stdout.write('y\n')
                elif util.negative_response(sys.stdin.readline()[:-1]):
                    exit(0)
                shutil.rmtree(args.summary_dir)
            os.makedirs(args.summary_dir)
        if args.report_file is not None:
            if os.path.isfile(args.report_file):
                sys.stdout.write('Delete file {file}? '.format(file=args.report_file))
                sys.stdout.flush()
                if args.yes:
                    sys.stdout.write('y\n')
                elif util.negative_response(sys.stdin.readline()[:-1]):
                    exit(0)
            report_file_dir = os.path.dirname(args.report_file)
            if report_file_dir and not os.path.isdir(report_file_dir):
                os.makedirs(report_file_dir)
            with open(args.report_file, 'w') as filehandle:
                filehandle.write('iteration,saved')
                for name in query:
                    filehandle.write(',train ' + name)
                if not args.tf_records:
                    for name in query:
                        filehandle.write(',validation ' + name)
                filehandle.write('\n')
    iteration_end = iteration_start + args.iterations - 1

    with Model(name=args.model, learning_rate=parameters.pop('learning_rate', 1e-3), weight_decay=parameters.pop('weight_decay', None), clip_gradients=parameters.pop('clip_gradients', None), model_directory=args.model_dir, summary_directory=args.summary_dir) as model:
        dropout = parameters.pop('dropout_rate', None)

        module = import_module('models.{}.{}'.format(dataset.type, args.model))
        if args.tf_records:
            inputs = tf_util.batch_records(dataset=dataset, mode='train', batch_size=args.batch_size)
            module.model(model=model, inputs=inputs, dataset_parameters=dataset_parameters, **parameters)
        else:
            module.model(model=model, inputs=dict(), dataset_parameters=dataset_parameters, **parameters)  # no input tensors, hence None for placeholder creation
        model.finalize(restore=args.restore)

        if args.verbosity >= 1:
            sys.stdout.write('         parameters: {:,}\n'.format(model.num_parameters))
            sys.stdout.write('         bytes: {:,}\n'.format(model.num_bytes))
            sys.stdout.write('{} train model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.write('         0%  {}/{}  '.format(iteration_start - 1, iteration_end))
            sys.stdout.flush()
        before = datetime.now()
        time_since_save = timedelta()

        if args.tf_records:
            train = {name: 0.0 for name in query}
            n = 0

            for iteration in range(iteration_start, iteration_end + 1):
                queried = model(query=query, optimize=True, summarize=True, dropout=dropout)
                train = {name: value + queried[name] for name, value in train.items()}
                n += 1

                if iteration % args.evaluation_frequency == 0 or (iteration < 5 * args.evaluation_frequency and iteration % args.evaluation_frequency == args.evaluation_frequency // 2) or iteration == 1 or iteration == iteration_end:
                    train = {name: value / n for name, value in train.items()}
                    after = datetime.now()
                    time_since_save += (after - before)

                    if args.report_file:
                        with open(args.report_file, 'a') as filehandle:
                            filehandle.write(str(iteration))
                            if args.model_dir is not None and (time_since_save.seconds > args.save_frequency * 60 * 60 or iteration == iteration_end):
                                filehandle.write(',yes')
                            else:
                                filehandle.write(',no')
                            for name in query:
                                filehandle.write(',' + str(train[name]))
                            filehandle.write('\n')

                    if args.verbosity >= 1:
                        sys.stdout.write('\r         {:.0f}%  {}/{}  '.format((iteration - iteration_start + 1) * 100 / args.iterations, iteration, iteration_end))
                        for name in query:
                            sys.stdout.write('{}={:.3f}  '.format(name, train[name]))
                        sys.stdout.write('(time per evaluation iteration: {})'.format(str(after - before).split('.')[0]))

                    if args.model_dir is not None and (time_since_save.seconds > args.save_frequency * 60 * 60 or iteration == iteration_end):
                        model.save()
                        if args.verbosity >= 1:
                            sys.stdout.write(' (model saved)')
                            sys.stdout.flush()
                        time_since_save = timedelta()
                    elif args.verbosity >= 1:
                        # sys.stdout.write('              ')
                        sys.stdout.flush()
                    before = datetime.now()
                    train = {name: 0.0 for name in train}
                    n = 0

        else:
            for iteration in range(iteration_start, iteration_end + 1):
                generated = dataset.generate(n=args.batch_size, mode='train')
                model(data=generated, optimize=True, summarize=True, dropout=dropout)

                if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
                    train = {name: 0.0 for name in query}
                    validation = {name: 0.0 for name in query}

                    if args.evaluation_iterations > 0:
                        for _ in range(args.evaluation_iterations):
                            generated = dataset.generate(n=args.batch_size, mode='train')
                            queried = model(query=query, data=generated)
                            train = {name: value + queried[name] for name, value in train.items()}
                        train = {name: value / args.evaluation_iterations for name, value in train.items()}

                        for _ in range(args.evaluation_iterations):
                            generated = dataset.generate(n=args.batch_size, mode='validation')
                            queried = model(query=query, data=generated)
                            validation = {name: value + queried[name] for name, value in validation.items()}
                        validation = {name: value / args.evaluation_iterations for name, value in validation.items()}

                    after = datetime.now()
                    if args.report_file:
                        with open(args.report_file, 'a') as filehandle:
                            filehandle.write(str(iteration))
                            if args.model_dir is not None and (time_since_save.seconds > args.save_frequency * 60 * 60 or iteration == iteration_end):
                                filehandle.write(',yes')
                            else:
                                filehandle.write(',no')
                            for name in query:
                                filehandle.write(',' + str(train[name]))
                            for name in query:
                                filehandle.write(',' + str(validation[name]))
                            filehandle.write('\n')

                    if args.verbosity >= 1:
                        sys.stdout.write('\r         {:.0f}%  {}/{}  '.format((iteration - iteration_start + 1) * 100 / args.iterations, iteration, iteration_end))
                        sys.stdout.write('train: ')
                        for name in query:
                            sys.stdout.write('{}={:.3f} '.format(name, train[name]))
                        sys.stdout.write(' validation: ')
                        for name in query:
                            sys.stdout.write('{}={:.3f} '.format(name, validation[name]))
                        sys.stdout.write(' (time per evaluation iteration: {})'.format(str(after - before).split('.')[0]))

                    time_since_save += (after - before)
                    if args.model_dir is not None and (time_since_save.seconds > args.save_frequency * 60 * 60 or iteration == iteration_end):
                        model.save()
                        if args.verbosity >= 1:
                            sys.stdout.write(' (model saved)')
                            sys.stdout.flush()
                        time_since_save = timedelta()
                    elif args.verbosity >= 1:
                        sys.stdout.write('              ')
                        sys.stdout.flush()
                    before = datetime.now()

    if args.verbosity >= 1:
        sys.stdout.write('\n{} model training finished\n'.format(datetime.now().strftime('%H:%M:%S')))
        sys.stdout.flush()
