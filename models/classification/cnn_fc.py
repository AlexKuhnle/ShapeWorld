from math import sqrt
import tensorflow as tf


# model parameters
sizes = (5, 3, 3)
num_filters = (8, 16, 32)
poolings = ('max', 'max', 'max')
hidden_dims = (512,)
hidden_dropouts = (0.0,)
learning_rate = 0.001
batch_size = 128
evaluation_size = 1024


def model(world, classification, dropouts, mode=None):

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
        weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, classification.get_shape()[1].value), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
        prediction = tf.matmul(a=embedding, b=weights)

    with tf.name_scope(name='optimization'):
        if mode == 'count':
            tf.losses.mean_pairwise_squared_error(labels=classification, predictions=prediction)
            prediction = tf.round(x=prediction)
        elif mode == 'multi':
            tf.losses.sigmoid_cross_entropy(multi_class_labels=classification, logits=prediction)
            prediction = tf.cast(x=tf.greater(x=prediction, y=0.5), dtype=tf.float32)
        else:
            tf.losses.softmax_cross_entropy(onehot_labels=classification, logits=prediction)
            prediction = tf.one_hot(indices=tf.argmax(input=prediction, axis=1), depth=classification.get_shape()[1].value)
        relevant = tf.reduce_sum(input_tensor=classification, axis=1)
        selected = tf.reduce_sum(input_tensor=prediction, axis=1)
        true_positive = tf.reduce_sum(input_tensor=tf.minimum(x=prediction, y=classification), axis=1)
        precision = tf.reduce_mean(input_tensor=tf.divide(x=true_positive, y=selected), axis=0)
        recall = tf.reduce_mean(input_tensor=tf.divide(x=true_positive, y=relevant), axis=0)

    return precision, recall
