from math import sqrt
import tensorflow as tf


def model(model, inputs, dataset_parameters, cnn_size, cnn_depth, cnn_block_depth, embedding_size, mlp_size, mlp_depth):

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    assert cnn_sizes[-1] % 2 == 0
    lstm_size = cnn_sizes[-1] // 2
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    with tf.name_scope(name='cnn'):
        world = inputs.get('world')
        if world is None:
            world = tf.placeholder(dtype=tf.float32, shape=((None,) + dataset_parameters['world_shape']), name='world')
            model.register_placeholder(key='world', placeholder=world)
        for size, depth in zip(cnn_sizes, cnn_depths):
            for _ in range(depth):
                weights = tf.Variable(initial_value=tf.random_normal(shape=(3, 3, world.get_shape()[3].value, size), stddev=sqrt(2.0 / world.get_shape()[3].value)))
                world = tf.nn.conv2d(input=world, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
                bias = tf.Variable(initial_value=tf.zeros(shape=(size,)))
                world = tf.nn.bias_add(value=world, bias=bias)
                world = tf.nn.relu(features=world)
                world = tf.nn.max_pool(value=world, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        for _ in range(len(world.get_shape()) - 2):
            world = tf.reduce_max(input_tensor=world, axis=1)

    with tf.name_scope(name='lstm'):
        embeddings = tf.Variable(initial_value=tf.random_normal(shape=(dataset_parameters['vocabulary_size'], embedding_size), stddev=sqrt(2.0 / embedding_size)))
        caption = inputs.get('caption')
        if caption is None:
            caption = tf.placeholder(dtype=tf.int32, shape=((None,) + dataset_parameters['caption_shape']), name='caption')
            model.register_placeholder(key='caption', placeholder=caption)
        embeddings = tf.nn.embedding_lookup(params=embeddings, ids=caption)
        lstm = tf.contrib.rnn.LSTMCell(num_units=lstm_size)
        caption_length = inputs.get('caption_length')
        if caption_length is None:
            caption_length = tf.placeholder(dtype=tf.int32, shape=(None,), name='caption_length')
            model.register_placeholder(key='caption_length', placeholder=caption_length)
        embeddings, state = tf.nn.dynamic_rnn(cell=lstm, inputs=embeddings, sequence_length=caption_length, dtype=tf.float32)
        caption = tf.concat(values=state, axis=1)

    with tf.name_scope(name='multiplication'):
        embedding = tf.multiply(x=world, y=caption)

    with tf.name_scope(name='mlp'):
        for size in mlp_sizes:
            weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, size), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
            embedding = tf.matmul(a=embedding, b=weights)
            bias = tf.Variable(initial_value=tf.zeros(shape=(size,)))
            embedding = tf.nn.bias_add(value=embedding, bias=bias)
            embedding = tf.nn.relu(features=embedding)

    with tf.name_scope(name='agreement'):
        weights = tf.Variable(initial_value=tf.random_normal(shape=(embedding.get_shape()[1].value, 1), stddev=sqrt(2.0 / embedding.get_shape()[1].value)))
        prediction = tf.matmul(a=embedding, b=weights)

    with tf.name_scope(name='optimization'):
        agreement = inputs.get('agreement')
        if agreement is None:
            agreement = tf.placeholder(dtype=tf.float32, shape=(None,), name='agreement')
            model.register_placeholder(key='agreement', placeholder=agreement)
        prediction = (tf.tanh(x=prediction) + 1.0) / 2.0
        cross_entropy = -(agreement * tf.log(x=prediction + 1e-10) + (1.0 - agreement) * tf.log(x=1.0 - prediction + 1e-10))
        loss = tf.reduce_mean(input_tensor=cross_entropy)
        tf.losses.add_loss(loss=loss)
        prediction = tf.cast(x=tf.greater(x=prediction, y=tf.constant(value=0.5)), dtype=tf.float32)
        correct = tf.cast(x=tf.equal(x=prediction, y=agreement), dtype=tf.float32)
        accuracy = tf.reduce_mean(input_tensor=correct)
        model.register_tensor(key=('agreement_accuracy'), tensor=accuracy)

    return agreement
