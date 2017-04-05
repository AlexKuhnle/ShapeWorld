from math import sqrt
import tensorflow as tf

# model parameters
sizes = (5, 3, 3)
num_filters = (16, 32, 64)
poolings = ('max', 'max', 'avg')
embedding_size = 32
lstm_size = 64
hidden_dims = (512,)


def model(world, caption, caption_length, agreement, dropouts, vocabulary_size):

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
        embeddings = tf.Variable(initial_value=tf.random_normal(shape=(vocabulary_size, embedding_size), stddev=sqrt(embedding_size)))
        embeddings = tf.nn.embedding_lookup(params=embeddings, ids=caption)
        lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
        embeddings, state = tf.nn.dynamic_rnn(cell=lstm, inputs=embeddings, sequence_length=tf.squeeze(input=caption_length, axis=1), dtype=tf.float32)
        caption_embedding = embeddings[:, -1, :]

    with tf.name_scope(name='concatenation'):
        embedding = tf.concat(values=(world_embedding, caption_embedding), axis=1)

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
        weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, 1), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
        prediction = tf.matmul(a=embedding, b=weights)

    with tf.name_scope(name='optimization'):
        prediction = (tf.tanh(x=prediction) + 1.0) / 2.0
        cross_entropy = -(agreement * tf.log(x=prediction + 1e-10) + (1.0 - agreement) * tf.log(x=1.0 - prediction + 1e-10))
        tf.losses.add_loss(loss=tf.reduce_mean(input_tensor=cross_entropy))

        prediction = tf.cast(x=tf.greater(x=prediction, y=tf.constant(value=0.5)), dtype=tf.float32)
        correct = tf.cast(x=tf.equal(x=prediction, y=agreement), dtype=tf.float32)
        accuracy = tf.reduce_mean(input_tensor=correct)

    return accuracy
