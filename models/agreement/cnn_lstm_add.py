import argparse
from datetime import datetime
from math import sqrt
import sys
import tensorflow as tf
from shapeworld.dataset import Dataset


parser = argparse.ArgumentParser(description='Image caption agreement model')
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
num_filters = (16, 32, 64)
poolings = ('max', 'max', 'avg')
embedding_size = 32
lstm_size = 64
hidden_dims = (512,)
hidden_dropouts = (0.0,)
learning_rate = 0.0001
batch_size = 128
evaluation_size = 1024


# dataset
dataset = Dataset.from_config(config=args.config, dataset_type='agreement', dataset_name=args.name)
sys.stdout.write('{} Caption agreement dataset: {}\n'.format(datetime.now().strftime('%H:%M:%S'), dataset))


# model architecture
with tf.name_scope(name='inputs'):
    world = tf.placeholder(dtype=tf.float32, shape=([None] + list(dataset.world_shape)))
    caption = tf.placeholder(dtype=tf.int32, shape=([None] + list(dataset.text_shape)))
    caption_length = tf.placeholder(dtype=tf.int32, shape=(None, 1))
    agreement = tf.placeholder(dtype=tf.float32, shape=(None, 1))


with tf.name_scope(name='cnn'):
    world_embedding = world
    for size, num_filter, pooling in zip(sizes, num_filters, poolings):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(size, size, world_embedding.get_shape()[3].value, num_filter), stddev=sqrt(2.0 / world_embedding.get_shape()[3].value)))
        world_embedding = tf.nn.conv2d(input=world_embedding, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
        bias = tf.Variable(initial_value=tf.zeros(shape=(num_filter,)))
        world_embedding = tf.nn.bias_add(value=world_embedding, bias=bias)
        world_embedding = tf.nn.relu(features=world_embedding)
        if pooling == 'max':
            world_embedding = tf.nn.max_pool(value=world_embedding, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        elif pooling == 'avg':
            world_embedding = tf.nn.avg_pool(value=world_embedding, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    size = 1
    for dim in world_embedding.get_shape()[1:]:
        size *= dim.value
    world_embedding = tf.reshape(tensor=world_embedding, shape=(-1, size))


with tf.name_scope(name='lstm'):
    embeddings = tf.Variable(initial_value=tf.random_normal(shape=(dataset.vocabulary_size, embedding_size), stddev=sqrt(embedding_size)))
    embeddings = tf.nn.embedding_lookup(params=embeddings, ids=caption)
    lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
    embeddings, state = tf.nn.dynamic_rnn(cell=lstm, inputs=embeddings, sequence_length=tf.squeeze(input=caption_length, axis=1), dtype=tf.float32)
    caption_embedding = embeddings[:, -1, :]


with tf.name_scope(name='addition'):
    scale = tf.Variable(initial_value=tf.random_normal(shape=(lstm_size, world_embedding.get_shape()[1].value), stddev=sqrt(2.0 / (lstm_size + world_embedding.get_shape()[1].value))))
    caption_embedding = tf.matmul(a=caption_embedding, b=scale)
    embedding = tf.add(x=world_embedding, y=caption_embedding)


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
    weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, 1), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
    prediction = tf.matmul(a=embedding, b=weights)


with tf.name_scope(name='optimization'):
    prediction = (tf.tanh(x=prediction) + 1.0) / 2.0
    cross_entropy = -(agreement * tf.log(x=prediction + 1e-10) + (1.0 - agreement) * tf.log(x=1.0 - prediction + 1e-10))
    tf.losses.add_loss(loss=tf.reduce_mean(input_tensor=cross_entropy))

    prediction = tf.cast(x=tf.greater(x=prediction, y=tf.constant(value=0.5)), dtype=tf.float32)
    correct = tf.cast(x=tf.equal(x=prediction, y=agreement), dtype=tf.float32)
    accuracy = tf.reduce_mean(input_tensor=correct)

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
                filehandle.write('iteration,train loss,train accuracy,validation loss,validation accuracy\n')

    if args.test:
        generated = dataset.generate(n=evaluation_size, mode='train')
        feed_dict = {world: generated['world'], caption: generated['caption'], caption_length: generated['caption-length'], agreement: generated['agreement']}
        feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
        training_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
        generated = dataset.generate(n=evaluation_size, mode='validation')
        feed_dict = {world: generated['world'], caption: generated['caption'], caption_length: generated['caption-length'], agreement: generated['agreement']}
        feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
        validation_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
        generated = dataset.generate(n=evaluation_size, mode='test')
        feed_dict = {world: generated['world'], caption: generated['caption'], caption_length: generated['caption-length'], agreement: generated['agreement']}
        feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
        test_accuracy = session.run(fetches=accuracy, feed_dict=feed_dict)
        sys.stdout.write('{} training={:.3f}, validation={:.3f}, test={:.3f}\n'.format(datetime.now().strftime('%H:%M:%S'), training_accuracy, validation_accuracy, test_accuracy))

    else:
        for iteration in range(iteration_start, iteration_start + args.iterations):
            generated = dataset.generate(n=batch_size, mode='train')
            feed_dict = {world: generated['world'], caption: generated['caption'], caption_length: generated['caption-length'], agreement: generated['agreement']}
            feed_dict.update(zip(dropouts, hidden_dropouts))
            session.run(fetches=optimization, feed_dict=feed_dict)
            if iteration % args.evaluation_frequency == 0 or iteration == 1 or iteration == args.evaluation_frequency // 2:
                generated = dataset.generate(n=evaluation_size, mode='train')
                feed_dict = {world: generated['world'], caption: generated['caption'], caption_length: generated['caption-length'], agreement: generated['agreement']}
                feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
                training_loss, training_accuracy = session.run(fetches=(tf.losses.get_total_loss(), accuracy), feed_dict=feed_dict)
                generated = dataset.generate(n=evaluation_size, mode='validation')
                feed_dict = {world: generated['world'], caption: generated['caption'], caption_length: generated['caption-length'], agreement: generated['agreement']}
                feed_dict.update(zip(dropouts, (0.0 for _ in range(len(dropouts)))))
                validation_loss, validation_accuracy = session.run(fetches=(tf.losses.get_total_loss(), accuracy), feed_dict=feed_dict)
                sys.stdout.write('{} Iteration {}: training={:.3f}, validation={:.3f}\n'.format(datetime.now().strftime('%H:%M:%S'), iteration, training_accuracy, validation_accuracy))
                if args.csv_file:
                    with open(args.csv_file, 'a') as filehandle:
                        filehandle.write('{},{},{},{},{}\n'.format(iteration, training_loss, training_accuracy, validation_loss, validation_accuracy))
            if args.model_file and iteration % args.save_frequency == 0:
                saver.save(session, args.model_file)
                sys.stdout.write('{} Model saved.\n'.format(datetime.now().strftime('%H:%M:%S')))
        sys.stdout.write('{} Training finished!\n'.format(datetime.now().strftime('%H:%M:%S')))
