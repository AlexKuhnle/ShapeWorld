import argparse
from datetime import datetime
from importlib import import_module
import os
import sys
from shapeworld import dataset, util


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate')

    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
    parser.add_argument('-c', '--config', type=util.parse_config, help='Dataset configuration file')

    parser.add_argument('-m', '--model', default='cnn_lstm_mult', help='Model')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-w', '--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('-d', '--dropout-rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('-p', '--parameters', type=util.parse_config, default=None, help='Model parameters')

    parser.add_argument('-b', '--batch-size', type=util.parse_int_with_factor, default=128, help='Batch size')
    parser.add_argument('-i', '--iterations', type=util.parse_int_with_factor, default=1000, help='Iterations')
    parser.add_argument('-e', '--evaluation-size', type=util.parse_int_with_factor, default=1024, help='Evaluation batch size')
    parser.add_argument('-v', '--evaluation-frequency', type=util.parse_int_with_factor, default=100, help='Evaluation frequency')

    parser.add_argument('-f', '--model-file', default=None, help='Model file')
    parser.add_argument('-r', '--report-file', default=None, help='CSV file reporting the evaluation results')
    parser.add_argument('-R', '--restore', action='store_true', help='Restore model (requires --model-file)')
    parser.add_argument('-E', '--evaluate', action='store_true', help='Evaluate model without training (requires --model-file)')
    parser.add_argument('-T', '--tf-records', action='store_true', help='TensorFlow queue with records (not compatible with --evaluate)')
    parser.add_argument('-V', '--verbose-tensorflow', action='store_true', help='TensorFlow verbosity')
    args = parser.parse_args()

    # import tensorflow
    if args.verbose_tensorflow:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    # import tf_util for TFRecords
    if args.tf_records:
        from shapeworld import tf_util
        assert not args.evaluate

    # information about dataset and model
    sys.stdout.write('{time} {dtype} dataset: {name}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=args.type, name=args.name))
    sys.stdout.write('         config: {config}\n'.format(config=args.config))
    sys.stdout.write('{time} {dtype} model: {model}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=args.type, model=args.model))
    sys.stdout.write('         parameters: {params}\n'.format(params=args.parameters))
    sys.stdout.flush()

    dataset = dataset(dtype=args.type, name=args.name, config=args.config)
    module = import_module('models.{}.{}'.format(args.type, args.model))

    if args.type == 'agreement':
        if args.tf_records:
            batch = tf_util.batch_records(paths=dataset.parts['tf-records'], dataset=dataset, batch_size=args.batch_size, noise=dataset.noise_range)
            world = batch['world']
            caption = batch['caption']
            caption_length = batch['caption_length']
            agreement = batch['agreement']
            dropout = tf.placeholder(dtype=tf.float32, shape=())
        else:
            with tf.name_scope(name='inputs'):
                world = tf.placeholder(dtype=tf.float32, shape=((None,) + dataset.world_shape))
                caption = tf.placeholder(dtype=tf.int32, shape=((None,) + dataset.vector_shape('caption')))
                caption_length = tf.placeholder(dtype=tf.int32, shape=(None, 1))
                agreement = tf.placeholder(dtype=tf.float32, shape=(None, 1))
                dropout = tf.placeholder(dtype=tf.float32, shape=())
            feed_dict_assignment = dict(world=world, caption=caption, caption_length=caption_length, agreement=agreement)
        parameters = args.parameters or dict()
        accuracy = module.model(world=world, caption=caption, caption_length=caption_length, agreement=agreement, dropout=dropout, vocabulary_size=dataset.vocabulary_size, **parameters)

    elif args.type == 'classification':
        with tf.name_scope(name='inputs'):
            world = tf.placeholder(dtype=tf.float32, shape=((None,) + dataset.world_shape))
            classification = tf.placeholder(dtype=tf.float32, shape=(None,) + dataset.vector_shape('classification'))
            dropout = tf.placeholder(dtype=tf.float32, shape=())
        feed_dict_assignment = dict(world=world, classification=classification)
        mode = None
        precision, recall = module.model(world=world, classification=classification, dropout=dropout, mode=mode)
        accuracy = recall

    else:
        assert False

    with tf.name_scope(name='optimization'):
        if args.weight_decay:
            for variable in tf.trainable_variables():
                weight_decay_loss = args.weight_decay * tf.nn.l2_loss(t=variable)
                tf.losses.add_loss(loss=weight_decay_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        loss = tf.losses.get_total_loss()
        grads_and_vars = optimizer.compute_gradients(loss=loss)
        optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)


    def get_feed_dict(batch_size, mode, dropout_rate):
        if args.tf_records:
            feed_dict = dict()
        else:
            generated = dataset.generate(n=batch_size, mode=mode)
            feed_dict = {placeholder: generated[value] for value, placeholder in feed_dict_assignment.items()}
        feed_dict[dropout] = dropout_rate
        return feed_dict


    # tf session
    with tf.Session() as session:
        iteration_start = 1
        if args.model_file:
            saver = tf.train.Saver()
        if args.restore or args.evaluate:
            sys.stdout.write('{time} restore model...\n'.format(time=datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            assert args.model_file
            saver.restore(sess=session, save_path=args.model_file)
            if args.report_file:
                with open(args.report_file, 'r') as filehandle:
                    for line in filehandle:
                        value = line.split(',')[0]
                if value != 'iteration':
                    iteration_start = int(value) + 1
        else:
            # initialize
            sys.stdout.write('{time} initialize model...\n'.format(time=datetime.now().strftime('%H:%M:%S')))
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

        coordinator = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

        if args.evaluate:
            # evaluation
            sys.stdout.write('{time} evaluate model...\n'.format(time=datetime.now().strftime('%H:%M:%S')))
            sys.stdout.write('         ')
            sys.stdout.flush()
            feed_dict = get_feed_dict(batch_size=args.evaluation_size, mode='train', dropout_rate=0.0)
            train_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
            sys.stdout.write('train={accuracy:.3f}'.format(accuracy=train_accuracy))
            feed_dict = get_feed_dict(batch_size=args.evaluation_size, mode='validation', dropout_rate=0.0)
            validation_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
            sys.stdout.write('  validation={accuracy:.3f}'.format(accuracy=validation_accuracy))
            feed_dict = get_feed_dict(batch_size=args.evaluation_size, mode='test', dropout_rate=0.0)
            test_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
            sys.stdout.write('  test={accuracy:.3f}\n'.format(accuracy=test_accuracy))
            sys.stdout.write('{time} evaluation finished!\n'.format(time=datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()

        else:
            # training
            sys.stdout.write('{time} train model...\n'.format(time=datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            before = datetime.now()
            for iteration in range(iteration_start, iteration_end + 1):
                feed_dict = get_feed_dict(batch_size=args.batch_size, mode='train', dropout_rate=args.dropout_rate)
                session.run(fetches=optimization, feed_dict=feed_dict)
                if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2 or iteration == iteration_end:
                    feed_dict = get_feed_dict(batch_size=args.evaluation_size, mode='train', dropout_rate=0.0)
                    train_loss, train_accuracy = session.run(fetches=(loss, accuracy), feed_dict=feed_dict)
                    feed_dict = get_feed_dict(batch_size=args.evaluation_size, mode='validation', dropout_rate=0.0)
                    validation_loss, validation_accuracy = session.run(fetches=(loss, accuracy), feed_dict=feed_dict)
                    after = datetime.now()
                    sys.stdout.write('\r         {completed:.0f}%  {iteration}/{iterations}  train={train:.3f}  validation={validation:.3f}  (time per evaluation iteration: {duration})'.format(completed=(iteration * 100 / iteration_end), iteration=iteration, iterations=iteration_end, train=train_accuracy, validation=validation_accuracy, duration=str(after - before).split('.')[0]))
                    sys.stdout.flush()
                    before = datetime.now()
                    if args.report_file:
                        with open(args.report_file, 'a') as filehandle:
                            filehandle.write('{},{},{},{},{}\n'.format(iteration, train_loss, train_accuracy, validation_loss, validation_accuracy))
            sys.stdout.write('\n{time} model training finished!\n'.format(time=datetime.now().strftime('%H:%M:%S')))
            sys.stdout.flush()
            if args.model_file:
                saver.save(sess=session, save_path=args.model_file)
                sys.stdout.write('{time} model saved.\n'.format(time=datetime.now().strftime('%H:%M:%S')))
                sys.stdout.flush()

        coordinator.request_stop()
        coordinator.join(threads=queue_threads)
