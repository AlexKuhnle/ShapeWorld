import argparse
from datetime import datetime, timedelta
from importlib import import_module
import json
import os
import shutil
import sys
from shapeworld import dataset, util
from models.TFMacros.tf_macros import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')

    parser.add_argument('-t', '--type', help='Dataset type')
    parser.add_argument('-n', '--name', help='Dataset name')
    parser.add_argument('-l', '--language', default=None, help='Dataset language')
    parser.add_argument('-c', '--config', type=util.parse_config, default=None, help='Dataset configuration file')
    parser.add_argument('-p', '--pixel-noise', type=float, default=0.1, help='Pixel noise range')

    parser.add_argument('-m', '--model', help='Model')
    parser.add_argument('-y', '--hyperparams-file', default=None, help='Model hyperparameters file (default: hyperparams directory)')
    parser.add_argument('-r', '--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('-R', '--restore', action='store_true', help='Restore model (requires --model-file)')

    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=1000, help='Iterations')
    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=100, help='Batch size')
    parser.add_argument('-e', '--evaluation-size', type=util.parse_int_with_factor, default=1000, help='Evaluation batch size')
    parser.add_argument('-f', '--evaluation-frequency', type=util.parse_int_with_factor, default=100, help='Evaluation frequency')
    parser.add_argument('-q', '--query', default=None, help='Additional values to query (separated by commas)')
    parser.add_argument('-T', '--tf-records', action='store_true', help='TensorFlow queue with records (not compatible with --evaluation-size)')

    parser.add_argument('--model-dir', default=None, help='TensorFlow model directory, storing the model computation graph and parameters')
    parser.add_argument('--summary-dir', default=None, help='TensorFlow summary directory for TensorBoard, reporting variable values etc')
    parser.add_argument('--report-file', default=None, help='CSV file reporting the training results')

    parser.add_argument('-v', '--verbosity', type=int, choices=(0, 1, 2), default=1, help='Verbosity (0: nothing, 1: default, 2: TensorFlow)')
    args = parser.parse_args()

    # import tensorflow
    if args.verbosity >= 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    # import tf_util for TFRecords
    if args.tf_records:
        from shapeworld import tf_util

    # dataset
    dataset = dataset(dtype=args.type, name=args.name, language=args.language, config=args.config)

    # information about dataset and model
    if args.verbosity >= 1:
        sys.stdout.write('{time} train {model} on {dataset}\n'.format(
            time=datetime.now().strftime('%H:%M:%S'),
            model=args.model,
            dataset=dataset
        ))
        sys.stdout.write('         config: {}\n'.format(args.config))
        sys.stdout.write('         hyperparameters: {}\n'.format(args.hyperparams_file))
        sys.stdout.flush()

    if args.type == 'agreement':
        parameters = dict(
            world_shape=dataset.world_shape,
            vocabulary_size=dataset.vocabulary_size(value_type='language'),
            caption_shape=dataset.vector_shape(value_name='caption')
        )
        query = ('agreement_accuracy',)

    elif args.type == 'classification':
        parameters = dict(
            world_shape=dataset.world_shape,
            num_classes=dataset.num_classes,
            multi_class=dataset.multi_class,
            class_count=dataset.class_count
        )
        query = ('classification_fscore', 'classification_precision', 'classification_recall')

    elif args.type == 'clevr_classification':
        parameters = dict(
            world_shape=dataset.world_shape,
            vocabulary_size=dataset.vocabulary_size,
            question_shape=dataset.vector_shape('question'),
            num_answers=len(dataset.answers)
        )
        query = ('answer_fscore', 'answer_precision', 'answer_recall')

    else:
        assert False

    query += ('loss',)
    if args.query:
        query += tuple(args.query.split(','))

    if args.hyperparams_file is None:
        with open(os.path.join('models', dataset.type, 'hyperparams', args.model + '.params.json'), 'r') as filehandle:
            parameters.update(json.load(fp=filehandle))
    else:
        with open(args.hyperparams_file, 'r') as filehandle:
            parameters.update(json.load(fp=filehandle))
    if args.learning_rate is not None:
        parameters['learning_rate'] = args.learning_rate

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
        if args.model_dir:
            if os.path.isdir(args.model_dir):
                shutil.rmtree(args.model_dir)
            os.makedirs(args.model_dir)
        if args.summary_dir:
            if os.path.isdir(args.summary_dir):
                shutil.rmtree(args.summary_dir)
            os.makedirs(args.summary_dir)
        if args.report_file:
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

    with Model(name=args.model, learning_rate=parameters.pop('learning_rate'), weight_decay=parameters.pop('weight_decay', 0.0), model_directory=args.model_dir, summary_directory=args.summary_dir) as model:
        dropout = parameters.pop('dropout_rate', 0.0)

        module = import_module('models.{}.{}'.format(args.type, args.model))
        if args.tf_records:
            module.model(model=model, inputs=tf_util.batch_records(dataset=dataset, batch_size=args.batch_size, noise_range=args.pixel_noise), **parameters)
        else:
            module.model(model=model, inputs=dict(), **parameters)  # no input tensors, hence None for placeholder creation
        model.finalize(restore=args.restore)

        if args.verbosity >= 1:
            sys.stdout.write('         parameters: {:,}\n'.format(model.num_parameters))
            sys.stdout.write('         bytes: {:,}\n'.format(model.num_bytes))
            sys.stdout.write('{} train model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.write('         0%  {}/{}  '.format(iteration_start - 1, iteration_end))
            sys.stdout.flush()
        before = datetime.now()
        time_since_save = timedelta()
        save_frequency = 60 * 60 * 3  # 3 hours

        if args.tf_records:
            train = {name: 0.0 for name in query}
            n = 0
            for iteration in range(iteration_start, iteration_end + 1):
                queried = model(query=query, optimize=True, dropout=dropout)  # loss !!!???
                train = {name: value + queried[name] for name, value in train.items()}
                n += 1
                if iteration % args.evaluation_frequency == 0 or (iteration < 5 * args.evaluation_frequency and iteration % args.evaluation_frequency == args.evaluation_frequency // 2) or iteration == 1 or iteration == iteration_end:
                    train = {name: value / n for name, value in train.items()}
                    after = datetime.now()
                    time_since_save += (after - before)
                    if args.report_file:
                        with open(args.report_file, 'a') as filehandle:
                            filehandle.write(str(iteration))
                            if time_since_save.seconds > save_frequency or iteration == iteration_end:
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
                    if time_since_save.seconds > save_frequency or iteration == iteration_end:
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
                generated = dataset.generate(n=args.batch_size, mode='train', noise_range=args.pixel_noise)
                model(data=generated, optimize=True, dropout=dropout)
                if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
                    generated = dataset.generate(n=args.evaluation_size, mode='train', noise_range=args.pixel_noise)
                    train = model(query=query, data=generated)
                    generated = dataset.generate(n=args.evaluation_size, mode='validation', noise_range=args.pixel_noise)
                    validation = model(query=query, data=generated)
                    after = datetime.now()
                    if args.report_file:
                        with open(args.report_file, 'a') as filehandle:
                            filehandle.write(str(iteration))
                            if time_since_save.seconds > save_frequency or iteration == iteration_end:
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
                    if time_since_save.seconds > save_frequency or iteration == iteration_end:
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
