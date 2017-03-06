import argparse
from datetime import datetime
from math import sqrt
import sys
import tensorflow as tf
from shapeworld.dataset import Dataset


parser = argparse.ArgumentParser(description='Classification model')
parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
parser.add_argument('-c', '--config', default=None, help='Dataset configuration file')
parser.add_argument('-i', '--iterations', type=int, default=1000, help='Iterations')
parser.add_argument('-e', '--evaluation-frequency', type=int, default=100, help='Evaluation frequency')
parser.add_argument('-s', '--save-frequency', type=int, default=1000, help='Save frequency')
parser.add_argument('-m', '--model-file', default=None, help='Model file')
parser.add_argument('-v', '--csv-file', default=None, help='CSV file reporting the evaluation results')
parser.add_argument('-r', '--restore', action='store_true', help='Restore model')
parser.add_argument('-t', '--test', action='store_true', help='Test model without training (use with --restore)')
args = parser.parse_args()


# model parameters
sizes = (5, 3, 3)
num_filters = (8, 16, 32)
poolings = ('max', 'max', 'max')
hidden_dims = (512,)
hidden_dropouts = (0.0,)
learning_rate = 0.001
batch_size = 128
evaluation_size = 1024


# dataset
dataset = Dataset.from_config(config=args.config, dataset_type='classification', dataset_name=args.name)
sys.stdout.write('{} Classification dataset: {}\n'.format(datetime.now().strftime('%H:%M:%S'), dataset))


# model architecture
with tf.name_scope(name='inputs'):
    world = tf.placeholder(dtype=tf.float32, shape=([None] + list(dataset.world_shape)))
    classification = tf.placeholder(dtype=tf.float32, shape=(None, dataset.num_classes))


with tf.name_scope(name='cnn'):
    embedding = world
    for size, num_filter, pooling in zip(sizes, num_filters, poolings):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(size, size, embedding.get_shape()[3].value, num_filter), stddev=sqrt(2.0 / embedding.get_shape()[3].value)))
        embedding = tf.nn.conv2d(input=embedding, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
        bias = tf.Variable(initial_value=tf.zeros(shape=(num_filter,)))
        embedding = tf.nn.bias_add(value=embedding, bias=bias)
        embedding = tf.nn.relu(features=embedding)
        if pooling == 'max':
            embedding = tf.nn.max_pool(value=embedding, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        elif pooling == 'avg':
            embedding = tf.nn.avg_pool(value=embedding, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    size = 1
    for dim in embedding.get_shape()[1:]:
        size *= dim.value
    embedding = tf.reshape(tensor=embedding, shape=(-1, size))


with tf.name_scope(name='hidden'):
    dropouts = []
    for dim in hidden_dims:
        weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, dim), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
        embedding = tf.matmul(a=embedding, b=weights)
        bias = tf.Variable(initial_value=tf.zeros(shape=(dim,)))
        embedding = tf.nn.bias_add(value=embedding, bias=bias)
        embedding = tf.nn.relu(features=embedding)
        dropout = tf.placeholder(dtype=tf.float32, shape=())
        dropouts.append(dropout)
        embedding = tf.nn.dropout(x=embedding, keep_prob=(1.0 - dropout))


with tf.name_scope(name='agreement'):
    weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, dataset.num_classes), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
    prediction = tf.matmul(a=embedding, b=weights)


with tf.name_scope(name='optimization'):
    if dataset.__class__.class_count_flag:
        tf.losses.mean_pairwise_squared_error(labels=classification, predictions=prediction)
        prediction = tf.round(x=prediction)
    elif dataset.__class__.multi_class_flag:
        tf.losses.sigmoid_cross_entropy(multi_class_labels=classification, logits=prediction)
        prediction = tf.cast(x=tf.greater(x=prediction, y=0.5), dtype=tf.float32)
    else:
        tf.losses.softmax_cross_entropy(onehot_labels=classification, logits=prediction)
        prediction = tf.one_hot(indices=tf.argmax(input=prediction, axis=1), depth=dataset.num_classes)

    relevant = tf.reduce_sum(input_tensor=classification, axis=1)
    selected = tf.reduce_sum(input_tensor=prediction, axis=1)
    true_positive = tf.reduce_sum(input_tensor=tf.minimum(x=prediction, y=classification), axis=1)
    precision = tf.reduce_mean(input_tensor=tf.divide(x=true_positive, y=selected), axis=0)
    recall = tf.reduce_mean(input_tensor=tf.divide(x=true_positive, y=relevant), axis=0)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss=tf.losses.get_total_loss())
    optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)


with tf.Session() as session:
    iteration_start = 1
    if args.model_file:
        saver = tf.train.Saver()
    if args.restore or args.test:
        assert args.model_file
        saver.restore(session, args.model_file)
        sys.stdout.write('{} Model restored.\n'.format(datetime.now().strftime('%H:%M:%S')))
        if args.csv_file:
            with open(args.csv_file, 'r') as filehandle:
                for line in filehandle:
                    value = line.split(',')[0]
                    if value != 'iteration':
                        iteration_start = int(value) + 1
    else:
        session.run(fetches=tf.global_variables_initializer())
        sys.stdout.write('{} Model initialized.\n'.format(datetime.now().strftime('%H:%M:%S')))
        if args.csv_file:
            with open(args.csv_file, 'w') as filehandle:
                filehandle.write('iteration,loss,accuracy\n')

    if args.test:
        generated = dataset.generate(n=evaluation_size)
        feed_dict = {world: generated['world'], classification: generated['class']}
        feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
        current_precision, current_recall = session.run(fetches=(precision, recall), feed_dict=feed_dict)
        sys.stdout.write('{} precision={:.3f}, recall={:.3f}\n'.format(datetime.now().strftime('%H:%M:%S'), current_precision, current_recall))

    else:
        for iteration in range(iteration_start, iteration_start + args.iterations):
            generated = dataset.generate(n=batch_size)
            feed_dict = {world: generated['world'], classification: generated['classification']}
            feed_dict.update(zip(dropouts, hidden_dropouts))
            session.run(fetches=optimization, feed_dict=feed_dict)
            if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2:
                generated = dataset.generate(n=evaluation_size)
                feed_dict = {world: generated['world'], classification: generated['classification']}
                feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
                current_loss, current_precision, current_recall = session.run(fetches=(tf.losses.get_total_loss(), precision, recall), feed_dict=feed_dict)
                sys.stdout.write('{} Iteration {}: precision={:.3f}, recall={:.3f}\n'.format(datetime.now().strftime('%H:%M:%S'), iteration, current_precision, current_recall))
                if args.csv_file:
                    with open(args.csv_file, 'a') as filehandle:
                        filehandle.write('{},{},{}\n'.format(iteration, current_loss, current_precision, current_recall))
            if args.model_file and iteration % args.save_frequency == 0:
                saver.save(session, args.model_file)
                sys.stdout.write('{} Model saved.\n'.format(datetime.now().strftime('%H:%M:%S')))
        sys.stdout.write('{} Training finished!\n'.format(datetime.now().strftime('%H:%M:%S')))
