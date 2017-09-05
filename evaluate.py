import argparse
from datetime import datetime
from importlib import import_module
import os
import sys
from shapeworld import dataset, util
from models.TFMacros.tf_macros import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')

    parser.add_argument('-t', '--type', help='Dataset type')
    parser.add_argument('-n', '--name', help='Dataset name')
    parser.add_argument('-l', '--language', help='Dataset language')
    parser.add_argument('-c', '--config', type=util.parse_config, help='Dataset configuration file')
    parser.add_argument('-p', '--pixel-noise', type=float, default=0.1, help='Pixel noise range')

    parser.add_argument('-m', '--model', help='Model')
    parser.add_argument('-r', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-w', '--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('-d', '--dropout-rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('-y', '--hyperparameters', type=util.parse_config, default=None, help='Model hyperparameters')

    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=1000, help='Iterations')
    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=128, help='Batch size')
    parser.add_argument('-e', '--evaluation-size', type=util.parse_int_with_factor, default=1024, help='Evaluation batch size')
    parser.add_argument('-f', '--evaluation-frequency', type=util.parse_int_with_factor, default=100, help='Evaluation frequency')
    parser.add_argument('-R', '--restore', action='store_true', help='Restore model (requires --model-file)')
    parser.add_argument('-E', '--evaluate', action='store_true', help='Evaluate model without training (requires --model-file)')
    parser.add_argument('-T', '--tf-records', action='store_true', help='TensorFlow queue with records (not compatible with --evaluate)')

    parser.add_argument('--model-file', default=None, help='TensorFlow model file, storing the model computation graph and parameters')
    parser.add_argument('--summary-file', default=None, help='TensorFlow summary file for TensorBoard, reporting variable values etc')
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

    # import tf_util for TFRecords
    if args.tf_records:
        from shapeworld import tf_util
        assert not args.evaluate

    # dataset
    dataset = dataset(dtype=args.type, name=args.name, language=args.language, config=args.config)

    # information about dataset and model
    if args.verbosity >= 1:
        sys.stdout.write('{time} {dataset}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dataset=dataset))
        sys.stdout.write('         config: {}\n'.format(args.config))
        sys.stdout.write('         {} model: {}\n'.format(args.type, args.model))
        sys.stdout.write('         hyperparameters: {}\n'.format(args.hyperparameters))
        sys.stdout.flush()

    # tf records
    if args.tf_records:
        inputs = tf_util.batch_records(dataset=dataset, batch_size=args.batch_size, noise_range=args.pixel_noise)
    else:
        inputs = dict()  # no input tensors, hence None for placeholder creation

    if args.type == 'agreement':
        parameters = dict(
            world_shape=dataset.world_shape,
            vocabulary_size=dataset.vocabulary_size,
            caption_shape=dataset.vector_shape('caption')
        )
        query = ('agreement-accuracy',)
        serialize = ()

    elif args.type == 'classification':
        parameters = dict(
            world_shape=dataset.world_shape,
            num_classes=dataset.num_classes,
            multi_class=dataset.multi_class,
            class_count=dataset.class_count
        )
        query = ('classification-precision', 'classification-recall')
        serialize = ()

    elif args.type == 'clevr_classification':
        parameters = dict(
            world_shape=dataset.world_shape,
            vocabulary_size=dataset.vocabulary_size,
            question_shape=dataset.vector_shape('question'),
            num_answers=len(dataset.answers)
        )
        query = ('answer-precision', 'answer-recall')
        serialize = ()

    else:
        assert False

    if args.hyperparameters:
        parameters.update(args.hyperparameters)
    if args.query:
        query = tuple(args.query.split(','))
    if args.serialize:
        assert args.evaluate
        serialize = tuple(args.serialize.split(','))
        query += serialize
    if not args.evaluate:
        query += ('loss',)

    # restore
    iteration_start = 1
    if args.restore or args.evaluate:
        if args.report_file:
            with open(args.report_file, 'r') as filehandle:
                for line in filehandle:
                    value = line.split(',')[0]
            if value != 'iteration':
                iteration_start = int(value) + 1
    else:
        if args.model_file:
            model_file_dir = os.path.dirname(args.model_file)
            if not os.path.isdir(model_file_dir):
                os.makedirs(model_file_dir)
        if args.summary_file:
            summary_file_dir = os.path.dirname(args.summary_file)
            if not os.path.isdir(summary_file_dir):
                os.makedirs(summary_file_dir)
        if args.report_file:
            report_file_dir = os.path.dirname(args.report_file)
            if not os.path.isdir(report_file_dir):
                os.makedirs(report_file_dir)
            with open(args.report_file, 'w') as filehandle:
                filehandle.write('iteration\n')
                for name in query:
                    filehandle.write(',train ' + name)
                if not args.tf_records:
                    for name in query:
                        filehandle.write(',validation ' + name)
                filehandle.write('\n')
    iteration_end = iteration_start + args.iterations - 1

    with Model(name=args.model, learning_rate=args.learning_rate, weight_decay=args.weight_decay, model_file=args.model_file, summary_file=args.summary_file) as model:
        module = import_module('models.{}.{}'.format(args.type, args.model))
        module.model(model=model, inputs=inputs, **parameters)
        model.finalize(restore=(args.restore or args.evaluate))

        if args.verbosity >= 1:
            sys.stdout.write('         parameters: {:,}\n'.format(model.num_parameters))
            sys.stdout.write('         bytes: {:,}\n'.format(model.num_bytes))
            sys.stdout.flush()

        if args.evaluate:
            if args.verbosity >= 1:
                sys.stdout.write('{} evaluate model...\n'.format(datetime.now().strftime('%H:%M:%S')))
                sys.stdout.write('         ')
                sys.stdout.flush()

            generated = dataset.generate(n=args.evaluation_size, mode='train', noise_range=args.pixel_noise)
            train = model(query=query, data=generated)
            if args.verbosity >= 1:
                sys.stdout.write('train: ')
                for name in query:
                    sys.stdout.write('{}={:.3f} '.format(name, train[name]))
            if serialize:
                dataset.serialize(path='/home/aok25/test2/train', generated=generated, additional={name: (train[name], serialize[name]) for name in serialize})

            generated = dataset.generate(n=args.evaluation_size, mode='validation', noise_range=args.pixel_noise)
            validation = model(query=query, data=generated)
            if args.verbosity >= 1:
                sys.stdout.write(' validation: ')
                for name in query:
                    sys.stdout.write('{}={:.3f} '.format(name, validation[name]))
            if serialize:
                dataset.serialize(path='/home/aok25/test2/validation', generated=generated, additional={name: (validation[name], serialize[name]) for name in serialize})

            generated = dataset.generate(n=args.evaluation_size, mode='test', noise_range=args.pixel_noise)
            test = model(query=query, data=generated)
            if args.verbosity >= 1:
                sys.stdout.write(' test: ')
                for name in query:
                    sys.stdout.write('{}={:.3f} '.format(name, test[name]))
            if serialize:
                dataset.serialize(path='/home/aok25/test2/test', generated=generated, additional={name: (test[name], serialize[name]) for name in serialize})

            if args.verbosity >= 1:
                sys.stdout.write('\n')
                sys.stdout.write('{} evaluation finished!\n'.format(datetime.now().strftime('%H:%M:%S')))
                sys.stdout.flush()

        else:  # training
            if args.verbosity >= 1:
                sys.stdout.write('{} train model...\n'.format(datetime.now().strftime('%H:%M:%S')))
                sys.stdout.flush()
            before = datetime.now()

            if args.tf_records:
                mean = {name: 0.0 for name in query}
                n = 0
                for iteration in range(iteration_start, iteration_end + 1):
                    train = model(query=query, optimize=True, dropout=args.dropout_rate)  # loss !!!???
                    mean = {name: value + train[name] for name, value in mean.items()}
                    n += 1
                    if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
                        mean = {name: value / n for name, value in mean.items()}
                        after = datetime.now()
                        if args.verbosity >= 1:
                            sys.stdout.write('\r         {:.0f}%  {}/{}  '.format(iteration * 100 / iteration_end, iteration, iteration_end))
                            for name in query:
                                sys.stdout.write('{}={:.3f}  '.format(name, train[name]))
                            sys.stdout.write('(time per evaluation iteration: {})'.format(str(after - before).split('.')[0]))
                            sys.stdout.flush()
                        before = datetime.now()
                        if args.report_file:
                            with open(args.report_file, 'a') as filehandle:
                                filehandle.write(str(iteration))
                                for name in query:
                                    filehandle.write(',' + str(train[name]))
                                filehandle.write('\n')
                        mean = {name: 0.0 for name in mean}
                        n = 0

            else:
                for iteration in range(iteration_start, iteration_end + 1):
                    generated = dataset.generate(n=args.batch_size, mode='train', noise_range=args.pixel_noise)
                    model(data=generated, optimize=True, dropout=args.dropout_rate)
                    if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
                        generated = dataset.generate(n=args.evaluation_size, mode='train', noise_range=args.pixel_noise)
                        train = model(query=query, data=generated)
                        generated = dataset.generate(n=args.evaluation_size, mode='validation', noise_range=args.pixel_noise)
                        validation = model(query=query, data=generated)
                        after = datetime.now()
                        if args.verbosity >= 1:
                            sys.stdout.write('\r         {:.0f}%  {}/{}  '.format(iteration * 100 / iteration_end, iteration, iteration_end))
                            sys.stdout.write('train: ')
                            for name in query:
                                sys.stdout.write('{}={:.3f} '.format(name, train[name]))
                            sys.stdout.write(' validation: ')
                            for name in query:
                                sys.stdout.write('{}={:.3f} '.format(name, validation[name]))
                            sys.stdout.write(' (time per evaluation iteration: {})'.format(str(after - before).split('.')[0]))
                            sys.stdout.flush()
                        before = datetime.now()
                        if args.report_file:
                            with open(args.report_file, 'a') as filehandle:
                                filehandle.write(str(iteration))
                                for name in query:
                                    filehandle.write(',' + str(train[name]))
                                for name in query:
                                    filehandle.write(',' + str(validation[name]))
                                filehandle.write('\n')
            if args.verbosity >= 1:
                sys.stdout.write('\n{} model training finished!\n'.format(datetime.now().strftime('%H:%M:%S')))
                sys.stdout.flush()
