import tensorflow as tf


options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)  # NONE, ZLIB, GZIP


def read_record(serialized_record, dataset):
    features = dict()
    for value_name, value_type in dataset.values.items():
        if value_type == 'int':
            features[value_name] = tf.FixedLenFeature(shape=(1,), dtype=tf.int64)
        elif value_type == 'float':
            features[value_name] = tf.FixedLenFeature(shape=(1,), dtype=tf.float32)
        elif value_type == 'vector(int)' or value_type == 'text':
            features[value_name] = tf.FixedLenFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.int64)
        elif value_type == 'vector(float)':
            features[value_name] = tf.FixedLenFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.float32)
        elif value_type == 'world':
            features[value_name] = tf.FixedLenFeature(shape=dataset.world_shape, dtype=tf.float32)
        else:
            pass
    record = tf.parse_single_example(serialized=serialized_record, features=features)
    return record


def read_records(paths, dataset):
    paths = tuple(path + ('' if path.endswith('.tfrecords.gz') else '.tfrecords.gz') for path in paths)
    path_queue = tf.train.string_input_producer(string_tensor=paths, shuffle=True, capacity=len(paths))
    reader = tf.TFRecordReader(options=options)
    _, serialized_records = reader.read(queue=path_queue)
    records = read_record(serialized_record=serialized_records, dataset=dataset)
    return records


def batch_records(paths, dataset, batch_size, noise_range=0.0):
    records = read_records(paths=paths, dataset=dataset)
    batch = tf.train.shuffle_batch(tensors=records, batch_size=batch_size, capacity=(batch_size * 50), min_after_dequeue=(batch_size * 10), num_threads=1)
    if noise_range:
        for value_name, value_type in dataset.values.items():
            if value_type == 'world':
                world = batch[value_name]
                world += tf.truncated_normal(shape=((batch_size,) + dataset.world_shape), mean=0.0, stddev=noise_range)
                world = tf.clip_by_value(t=world, clip_value_min=0.0, clip_value_max=1.0)
                batch[value_name] = world
    return batch


def write_record(record, dataset):
    features = dict()
    for value_name, value_type in dataset.values.items():
        if value_type == 'model':
            continue
        value = record[value_name].flatten()
        if value_type == 'int':
            features[value_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        elif value_type == 'float':
            features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        elif value_type == 'vector(int)' or value_type == 'text':
            features[value_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        elif value_type == 'vector(float)':
            features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        elif value_type == 'world':
            features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        else:
            assert False
    record = tf.train.Example(features=tf.train.Features(feature=features))
    serialized_record = record.SerializeToString()
    return serialized_record


def write_records(path, records, dataset):
    num_records = len(next(iter(records.values())))
    with tf.python_io.TFRecordWriter(path=(path + '.tfrecords.gz'), options=options) as writer:
        for n in range(num_records):
            record = {value_name: value[n] for value_name, value in records.items()}
            serialized_record = write_record(record=record, dataset=dataset)
            writer.write(record=serialized_record)
