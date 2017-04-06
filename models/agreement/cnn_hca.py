from math import sqrt
import tensorflow as tf


def model(world, caption, caption_length, agreement, dropouts, vocabulary_size, sizes=(5, 3, 3), num_filters=(16, 32), poolings=('max', 'max', 'avg'), embedding_size=32, co_attention_type='parallel', attention_size=64, hidden_dims=(512,), **kwargs):
    # co_attention_type: 'parallel' or 'alternating'

    with tf.name_scope(name='cnn'):
        world_embedding = world
        for size, num_filter, pooling in zip(sizes, num_filters + (embedding_size,), poolings):
            weights = tf.Variable(initial_value=tf.random_normal(shape=(size, size, world_embedding.get_shape()[3].value, num_filter), stddev=sqrt(2.0 / world_embedding.get_shape()[3].value)))
            world_embedding = tf.nn.conv2d(input=world_embedding, filter=weights, strides=(1, 1, 1, 1), padding='SAME')
            bias = tf.Variable(initial_value=tf.zeros(shape=(num_filter,)))
            world_embedding = tf.nn.bias_add(value=world_embedding, bias=bias)
            world_embedding = tf.nn.relu(features=world_embedding)
            if pooling == 'max':
                world_embedding = tf.nn.max_pool(value=world_embedding, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
            elif pooling == 'avg':
                world_embedding = tf.nn.avg_pool(value=world_embedding, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        world_embedding = tf.reshape(tensor=world_embedding, shape=(-1, world_embedding.get_shape()[1].value * world_embedding.get_shape()[2].value, num_filter))

    with tf.name_scope(name='word-embeddings'):
        embeddings = tf.Variable(initial_value=tf.random_normal(shape=(vocabulary_size, embedding_size), stddev=sqrt(embedding_size)))
        word_embeddings = tf.nn.embedding_lookup(params=embeddings, ids=caption)

    with tf.name_scope(name='phrase-cnns'):
        filters = tf.Variable(initial_value=tf.truncated_normal(shape=(1, embedding_size, embedding_size), stddev=0.01))
        unigrams = tf.nn.conv1d(value=word_embeddings, filters=filters, stride=1, padding='SAME')
        filters = tf.Variable(initial_value=tf.truncated_normal(shape=(2, embedding_size, embedding_size), stddev=0.01))
        bigrams = tf.nn.conv1d(value=word_embeddings, filters=filters, stride=1, padding='SAME')
        filters = tf.Variable(initial_value=tf.truncated_normal(shape=(3, embedding_size, embedding_size), stddev=0.01))
        trigrams = tf.nn.conv1d(value=word_embeddings, filters=filters, stride=1, padding='SAME')
        phrase_embeddings = tf.stack(values=(unigrams, bigrams, trigrams), axis=3)
        phrase_embeddings = tf.reduce_max(input_tensor=phrase_embeddings, axis=3)
        phrase_embeddings = tf.tanh(x=phrase_embeddings)
        dropout = tf.placeholder(dtype=tf.float32, shape=())
        dropouts.append(dropout)
        phrase_embeddings = tf.nn.dropout(x=phrase_embeddings, keep_prob=(1.0 - dropout))

    with tf.name_scope(name='sentence-lstm'):
        lstm = tf.contrib.rnn.LSTMCell(num_units=embedding_size)
        sentence_embeddings, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=phrase_embeddings, sequence_length=tf.squeeze(input=caption_length, axis=1), dtype=tf.float32)

    with tf.name_scope(name='co-attention'):
        if co_attention_type == 'parallel':
            world_embedding_word, caption_embedding_word = parallel_attention(image_embeddings=world_embedding, text_embeddings=word_embeddings, attention_size=attention_size, dropouts=dropouts)
            world_embedding_phrase, caption_embedding_phrase = parallel_attention(image_embeddings=world_embedding, text_embeddings=phrase_embeddings, attention_size=attention_size, dropouts=dropouts)
            world_embedding_sentence, caption_embedding_sentence = parallel_attention(image_embeddings=world_embedding, text_embeddings=sentence_embeddings, attention_size=attention_size, dropouts=dropouts)
        elif attention == 'alternating':
            world_embedding_word, caption_embedding_word = alternating_attention(image_embeddings=world_embedding, text_embeddings=word_embeddings, attention_size=attention_size, dropouts=dropouts)
            world_embedding_phrase, caption_embedding_phrase = alternating_attention(image_embeddings=world_embedding, text_embeddings=phrase_embeddings, attention_size=attention_size, dropouts=dropouts)
            world_embedding_sentence, caption_embedding_sentence = alternating_attention(image_embeddings=world_embedding, text_embeddings=sentence_embeddings, attention_size=attention_size, dropouts=dropouts)
        else:
            assert False

    with tf.name_scope(name='combination'):
        # h^w = tanh(W_w * (q'^w + v'^w))
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=(embedding_size, embedding_size), stddev=0.01))
        embedding = tf.add(x=world_embedding_word, y=caption_embedding_word)
        dropout = tf.placeholder(dtype=tf.float32, shape=())
        dropouts.append(dropout)
        embedding = tf.nn.dropout(x=embedding, keep_prob=(1.0 - dropout))
        embedding = tf.matmul(a=embedding, b=weights)
        embedding_word = tf.tanh(x=embedding)
        # h^p = tanh(W_p * [(q'^p + v'^p), h^w])
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=(2 * embedding_size, embedding_size), stddev=0.01))
        embedding = tf.add(x=world_embedding_phrase, y=caption_embedding_phrase)
        embedding = tf.concat(values=(embedding, embedding_word), axis=1)
        dropout = tf.placeholder(dtype=tf.float32, shape=())
        dropouts.append(dropout)
        embedding = tf.nn.dropout(x=embedding, keep_prob=(1.0 - dropout))
        embedding = tf.matmul(a=embedding, b=weights)
        embedding_phrase = tf.tanh(x=embedding)
        # h^s = tanh(W_s * [(q'^s + v'^s), h^p])
        weights = tf.Variable(initial_value=tf.truncated_normal(shape=(2 * embedding_size, embedding_size), stddev=0.01))
        embedding = tf.add(x=world_embedding_sentence, y=caption_embedding_sentence)
        embedding = tf.concat(values=(embedding, embedding_phrase), axis=1)
        dropout = tf.placeholder(dtype=tf.float32, shape=())
        dropouts.append(dropout)
        embedding = tf.nn.dropout(x=embedding, keep_prob=(1.0 - dropout))
        embedding = tf.matmul(a=embedding, b=weights)
        embedding = tf.tanh(x=embedding)
        # # p = softmax(W_h * h^s)
        # weights = tf.Variable(initial_value=tf.truncated_normal(shape=(embedding_size, num_answers), stddev=0.01))
        # answer_dist_logits = tf.matmul(a=tf.nn.dropout(x=hidden_sentence, keep_prob=0.5), b=weights)

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


def parallel_attention(image_embeddings, text_embeddings, attention_size, dropouts):
    _, num_image_embeddings, image_embedding_size = image_embeddings.get_shape().as_list()
    _, num_text_embeddings, text_embedding_size = text_embeddings.get_shape().as_list()
    reduced_image_embeddings = tf.reshape(tensor=image_embeddings, shape=(-1, image_embedding_size))
    reduced_text_embeddings = tf.reshape(tensor=text_embeddings, shape=(-1, text_embedding_size))

    # C = tanh(Q^T * W_b * V)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(text_embedding_size, image_embedding_size), stddev=0.01))
    affinity_matrix = tf.matmul(a=reduced_image_embeddings, b=weights, transpose_b=True)
    affinity_matrix = tf.reshape(tensor=affinity_matrix, shape=(-1, num_image_embeddings, text_embedding_size))
    affinity_matrix = tf.matmul(a=text_embeddings, b=affinity_matrix, transpose_b=True)
    affinity_matrix = tf.tanh(x=affinity_matrix)

    # W_v * V
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(image_embedding_size, attention_size), stddev=0.01))
    transformed_image = tf.matmul(a=reduced_image_embeddings, b=weights)
    transformed_image = tf.reshape(tensor=transformed_image, shape=(-1, num_image_embeddings, attention_size))
    # W_q * Q
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(text_embedding_size, attention_size), stddev=0.01))
    transformed_text = tf.matmul(a=reduced_text_embeddings, b=weights)
    transformed_text = tf.reshape(tensor=transformed_text, shape=(-1, num_text_embeddings, attention_size))

    # H^v = tanh(W_v * V + (W_q * Q) * C)
    image_attention = tf.matmul(a=transformed_text, b=affinity_matrix, transpose_a=True)
    image_attention = tf.add(x=image_attention, y=transformed_image)
    image_attention = tf.tanh(x=image_attention)
    dropout = tf.placeholder(dtype=tf.float32, shape=())
    dropouts.append(dropout)
    image_attention = tf.nn.dropout(x=image_attention, keep_prob=(1.0 - dropout))
    # a^v = softmax(w_hv * H^v)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(attention_size, 1), stddev=0.01))
    image_attention = tf.reshape(tensor=image_attention, shape=(-1, attention_size))
    image_attention = tf.matmul(a=image_attention, b=weights)
    image_attention = tf.reshape(tensor=image_attention, shape=(-1, num_image_embeddings, 1))
    image_attention = tf.nn.softmax(logits=image_attention, dim=1)
    # v' = a^v * v
    image_attention = tf.multiply(x=image_embeddings, y=image_attention)
    image_attention = tf.reduce_sum(input_tensor=image_attention, axis=1)

    # H^q = tanh(W_q * Q + (W_v * V) * C^T)
    text_attention = tf.matmul(a=affinity_matrix, b=transformed_image)
    text_attention = tf.add(x=text_attention, y=transformed_text)
    text_attention = tf.tanh(x=text_attention)
    dropout = tf.placeholder(dtype=tf.float32, shape=())
    dropouts.append(dropout)
    text_attention = tf.nn.dropout(x=text_attention, keep_prob=(1.0 - dropout))
    # a^q = softmax(w_hq * H^q)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(attention_size, 1), stddev=0.01))
    text_attention = tf.reshape(tensor=text_attention, shape=(-1, attention_size))
    text_attention = tf.matmul(a=text_attention, b=weights)
    text_attention = tf.reshape(tensor=text_attention, shape=(-1, num_text_embeddings, 1))
    text_attention = tf.nn.softmax(logits=text_attention, dim=1)
    # q' = a^q * q
    text_attention = tf.multiply(x=text_embeddings, y=text_attention)
    text_attention = tf.reduce_sum(input_tensor=text_attention, axis=1)

    return image_attention, text_attention


def alternating_attention(image_embeddings, text_embeddings, attention_size, dropouts):
    _, num_image_embeddings, image_embedding_size = image_embeddings.get_shape().as_list()
    _, num_text_embeddings, text_embedding_size = text_embeddings.get_shape().as_list()
    reduced_image_embeddings = tf.reshape(tensor=image_embeddings, shape=(-1, image_embedding_size))
    reduced_text_embeddings = tf.reshape(tensor=text_embeddings, shape=(-1, text_embedding_size))

    # first iteration: text attention
    # H = tanh(W_q * Q)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(text_embedding_size, attention_size), stddev=0.01))
    text_attention = tf.matmul(a=reduced_text_embeddings, b=weights)
    text_attention = tf.tanh(x=text_attention)
    dropout = tf.placeholder(dtype=tf.float32, shape=())
    dropouts.append(dropout)
    text_attention = tf.nn.dropout(x=text_attention, keep_prob=(1.0 - dropout))
    # a^q = softmax(w_hq * H)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(attention_size, 1), stddev=0.01))
    text_attention = tf.matmul(a=text_attention, b=weights)
    text_attention = tf.reshape(tensor=text_attention, shape=(-1, num_text_embeddings, 1))
    text_attention = tf.nn.softmax(logits=text_attention, dim=1)
    # q' = a^q * q
    text_attention = tf.multiply(x=text_embeddings, y=text_attention)
    text_attention = tf.reduce_sum(input_tensor=text_attention, axis=1)

    # second iteration: image attention
    # H = tanh(W_v * V + W_g * g * 1)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(text_embedding_size, attention_size), stddev=0.01))
    image_attention = tf.matmul(a=text_attention, b=weights)
    image_attention = tf.tile(input=image_attention, multiples=(1, attention_size))
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(image_embedding_size, attention_size), stddev=0.01))
    image_attention = tf.matmul(a=reduced_image_embeddings, b=weights)
    image_attention = tf.add(x=image_attention, y=image_attention)
    image_attention = tf.tanh(x=image_attention)
    dropout = tf.placeholder(dtype=tf.float32, shape=())
    dropouts.append(dropout)
    image_attention = tf.nn.dropout(x=image_attention, keep_prob=(1.0 - dropout))
    # a^v = softmax(w_hv * H)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(attention_size, 1), stddev=0.01))
    image_attention = tf.matmul(a=image_attention, b=weights)
    image_attention = tf.reshape(tensor=image_attention, shape=(-1, num_image_embeddings, 1))
    image_attention = tf.nn.softmax(logits=image_attention, dim=1)
    # v' = a^v * v
    image_attention = tf.multiply(x=image_embeddings, y=image_attention)
    image_attention = tf.reduce_sum(input_tensor=image_attention, axis=1)

    # third iteration: text attention
    # H = tanh(W_q * Q + W_g * g * 1)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(image_embedding_size, attention_size), stddev=0.01))
    text_attention = tf.matmul(a=image_attention, b=weights)
    text_attention = tf.tile(input=text_attention, multiples=(1, attention_size))
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(text_embedding_size, attention_size), stddev=0.01))
    text_attention = tf.matmul(a=reduced_text_embeddings, b=weights)
    text_attention = tf.add(x=text_attention, y=text_attention)
    text_attention = tf.tanh(x=text_attention)
    dropout = tf.placeholder(dtype=tf.float32, shape=())
    dropouts.append(dropout)
    text_attention = tf.nn.dropout(x=text_attention, keep_prob=(1.0 - dropout))
    # a^q = softmax(w_hq * H)
    weights = tf.Variable(initial_value=tf.truncated_normal(shape=(attention_size, 1), stddev=0.01))
    text_attention = tf.matmul(a=text_attention, b=weights)
    text_attention = tf.reshape(tensor=text_attention, shape=(-1, num_text_embeddings, 1))
    text_attention = tf.nn.softmax(logits=text_attention, dim=1)
    # q' = a^q * q
    text_attention = tf.multiply(x=text_embeddings, y=text_attention)
    text_attention = tf.reduce_sum(input_tensor=text_attention, axis=1)

    return image_attention, text_attention
