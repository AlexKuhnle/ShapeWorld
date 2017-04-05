import argparse
from datetime import datetime
from importlib import import_module
import os
import sys
import tensorflow as tf
from shapeworld import Dataset, util


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate')

    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
    parser.add_argument('-c', '--config', default=None, help='Dataset configuration file')

    parser.add_argument('-m', '--model', default='cnn_lstm_mult', help='Model')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-d', '--dropout-rate', type=float, default=0.0, help='Dropout rate')

    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=128, help='Batch size')
    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=1000, help='Iterations')
    parser.add_argument('-e', '--evaluation-size', type=util.parse_int_with_factor, default=1024, help='Evaluation batch size')
    parser.add_argument('-v', '--evaluation-frequency', type=util.parse_int_with_factor, default=100, help='Evaluation frequency')

    parser.add_argument('-f', '--model-file', default=None, help='Model file')
    parser.add_argument('-r', '--report-file', default=None, help='CSV file reporting the evaluation results')
    parser.add_argument('-R', '--restore', action='store_true', help='Restore model (requires --model-file)')
    parser.add_argument('-E', '--evaluate', action='store_true', help='Evaluate model without training (requires --model-file)')
    args = parser.parse_args()

    # dataset
    dataset = Dataset.from_config(config=args.config, dataset_type=args.type, dataset_name=args.name)
    sys.stdout.write('{} {} dataset: {}\n'.format(datetime.now().strftime('%H:%M:%S'), dataset.type, dataset.name))

    # model
    module = import_module('models.{}.{}'.format(args.type, args.model))

    if args.type == 'agreement':
        with tf.name_scope(name='inputs'):
            world = tf.placeholder(dtype=tf.float32, shape=((None,) + dataset.world_shape))
            caption = tf.placeholder(dtype=tf.int32, shape=((None,) + dataset.text_shape))
            caption_length = tf.placeholder(dtype=tf.int32, shape=(None, 1))
            agreement = tf.placeholder(dtype=tf.float32, shape=(None, 1))
            dropouts = list()
        feed_dict_assignment = {'world': world, 'caption': caption, 'caption-length': caption_length, 'agreement': agreement}
        accuracy = module.model(world=world, caption=caption, caption_length=caption_length, agreement=agreement, dropouts=dropouts, vocabulary_size=dataset.vocabulary_size)

    elif args.typpe == 'classification':
        with tf.name_scope(name='inputs'):
            world = tf.placeholder(dtype=tf.float32, shape=((None,) + dataset.world_shape))
            classification = tf.placeholder(dtype=tf.float32, shape=(None, dataset.num_classes))
        feed_dict_assignment = {'world': world, 'classification': classification}
        if dataset.__class__.class_count_flag:
            mode = 'count'
        elif dataset.__class__.multi_class_flag:
            mode = 'multi'
        else:
            mode = None
        accuracy = module.model(world=world, classification=classification, dropouts=dropouts, mode=mode)

    with tf.name_scope(name='optimization'):
        loss = tf.losses.get_total_loss()
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss=tf.losses.get_total_loss())
        optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    # tf session
    with tf.Session() as session:
        iteration_start = 1
        if args.model_file:
            saver = tf.train.Saver()
        if args.restore or args.evaluate:
            sys.stdout.write('{} Restore model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            assert args.model_file
            saver.restore(session, args.model_file)
            if args.report_file:
                with open(args.report_file, 'r') as filehandle:
                    for line in filehandle:
                        value = line.split(',')[0]
                if value != 'iteration':
                    iteration_start = int(value) + 1
        else:
            # initialize
            sys.stdout.write('{} Initialize model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            session.run(fetches=tf.global_variables_initializer())
            if args.model_file:
                model_file_dir = os.path.dirname(args.model_file)
                if not os.path.isdir(model_file_dir):
                    os.makedirs(model_file_dir)
            if args.report_file:
                report_file_dir = os.path.dirname(args.report_file)
                if not os.path.isdir(report_file_dir):
                    os.makedirs(report_file_dir)
                with open(args.report_file, 'w') as filehandle:
                    filehandle.write('iteration,train loss,train accuracy,validation loss,validation accuracy\n')
        iteration_end = iteration_start + args.iterations - 1

        if args.evaluate:
            # evaluation
            sys.stdout.write('{} Evaluate model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.write('         ')
            sys.stdout.flush()
            generated = dataset.generate(n=args.evaluation_size, mode='train')
            feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
            feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
            training_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
            sys.stdout.write('training={:.3f}'.format(training_accuracy))
            generated = dataset.generate(n=args.evaluation_size, mode='validation')
            feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
            feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
            validation_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
            sys.stdout.write('  validation={:.3f}'.format(validation_accuracy))
            generated = dataset.generate(n=args.evaluation_size, mode='test')
            feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
            feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
            test_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
            sys.stdout.write('  test={:.3f}\n'.format(test_accuracy))
            sys.stdout.write('{} Evaluation finished!\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()

        else:
            # training
            sys.stdout.write('{} Train model...\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            for iteration in range(iteration_start, iteration_end + 1):
                before = datetime.now()
                generated = dataset.generate(n=args.batch_size, mode='train')
                feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
                feed_dict.update(zip(dropouts, (args.dropout_rate for _ in range(len(dropouts)))))
                session.run(fetches=optimization, feed_dict=feed_dict)
                after = datetime.now()
                if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
                    generated = dataset.generate(n=args.evaluation_size, mode='train')
                    feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
                    feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
                    training_loss, training_accuracy = session.run(fetches=(loss, accuracy), feed_dict=feed_dict)
                    generated = dataset.generate(n=args.evaluation_size, mode='validation')
                    feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
                    feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
                    validation_loss, validation_accuracy = session.run(fetches=(loss, accuracy), feed_dict=feed_dict)
                    sys.stdout.write('\r         {:.0f}%  {}/{}  training={:.3f}  validation={:.3f}  (time per batch: {})'.format(iteration * 100 / iteration_end, iteration, iteration_end, training_accuracy, validation_accuracy, str(after - before).split('.')[0]))
                    sys.stdout.flush()
                    if args.report_file:
                        with open(args.report_file, 'a') as filehandle:
                            filehandle.write('{},{},{},{},{}\n'.format(iteration, training_loss, training_accuracy, validation_loss, validation_accuracy))
            sys.stdout.write('\n{} Model training finished!\n'.format(datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            if args.model_file:
                saver.save(session, args.model_file)
                sys.stdout.write('{} Model saved.\n'.format(datetime.now().strftime('%H:%M:%S')))
                sys.stdout.flush()
