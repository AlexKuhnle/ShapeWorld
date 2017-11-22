import tensorflow as tf
from shapeworld.dataset import alternatives_type


options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)


def read_record(dataset, serialized_record):
    features = dict()
    feature_lists = dict()
    for value_name, value_type in dataset.values.items():
        value_type, alts = alternatives_type(value_type=value_type)
        if value_type == 'int':
            if alts:
                feature_lists[value_name] = tf.FixedLenSequenceFeature(shape=(), dtype=tf.int64)
            else:
                features[value_name] = tf.FixedLenFeature(shape=(), dtype=tf.int64)
        elif value_type == 'float':
            if alts:
                feature_lists[value_name] = tf.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
            else:
                features[value_name] = tf.FixedLenFeature(shape=(), dtype=tf.float32)
        elif value_type == 'vector(int)' or dataset.vocabulary(value_type=value_type) is not None:
            if alts:
                feature_lists[value_name] = tf.FixedLenSequenceFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.int64)
            else:
                features[value_name] = tf.FixedLenFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.int64)
        elif value_type == 'vector(float)':
            if alts:
                feature_lists[value_name] = tf.FixedLenSequenceFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.float32)
            else:
                features[value_name] = tf.FixedLenFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.float32)
        elif value_type == 'world':
            features[value_name] = tf.FixedLenFeature(shape=dataset.world_shape, dtype=tf.float32)
        else:
            pass
    record = tf.parse_single_sequence_example(serialized=serialized_record, context_features=features, sequence_features=feature_lists)
    return record


def read_records(dataset):
    paths = tuple(path + ('' if path.endswith('.tfrecords.gz') else '.tfrecords.gz') for path in dataset.get_records_paths())
    path_queue = tf.train.string_input_producer(string_tensor=paths, shuffle=True, capacity=len(paths))
    reader = tf.TFRecordReader(options=options)
    _, serialized_records = reader.read(queue=path_queue)
    records = read_record(dataset=dataset, serialized_record=serialized_records)
    return records


def batch_records(dataset, batch_size, noise_range=None):
    records, sequence_records = read_records(dataset=dataset)
    if 'alternatives' in records:
        sample = tf.cast(x=tf.floor(x=tf.multiply(x=tf.cast(x=records['alternatives'], dtype=tf.float32), y=tf.random_uniform(shape=()))), dtype=tf.int32)
        for value_name, record in sequence_records.items():
            records[value_name] = sequence_records[value_name][sample]
            records[value_name] = tf.Print(records[value_name], (sample, records[value_name]))
    batch = tf.train.shuffle_batch(tensors=records, batch_size=batch_size, capacity=(batch_size * 50), min_after_dequeue=(batch_size * 10), num_threads=1)
    # if alternatives and 'alternatives' in batch:
    #     sample = tf.floor(x=tf.multiply(x=batch['alternatives'], y=tf.random_uniform(shape=(batch_size,))))
    for value_name, value_type in dataset.values.items():
        # value_type, alts = alternatives_type(value_type=value_type)
        # if not alternatives and alts:
        #     indices = tf.stack(values=(tf.range(limit=batch[value_name].shape[1].value), sample), axis=1)
        #     print(indices.get_shape().as_list())
        #     exit(0)
        #     batch[value_name] = tf.gather_nd(params=batch[value_name], indices=indices)
        #     print(batch[value_name].get_shape().as_list())
        #     exit(0)
        if noise_range is not None and noise_range > 0.0 and value_type == 'world':
            world = batch[value_name]
            noise = tf.truncated_normal(shape=((batch_size,) + dataset.world_shape), mean=0.0, stddev=noise_range)
            world = tf.clip_by_value(t=(world + noise), clip_value_min=0.0, clip_value_max=1.0)
            batch[value_name] = world
    if 'alternatives' in batch:
        batch.pop('alternatives')
    return batch


def write_record(dataset, record):
    features = dict()
    feature_lists = dict()
    for value_name, value_type in dataset.values.items():
        value_type, alts = alternatives_type(value_type=value_type)
        if value_type == 'model':
            continue
        if value_type == 'int':
            if alts:
                feature_lists[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=(value,))) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=(record[value_name],)))
        elif value_type == 'float':
            if alts:
                feature_lists[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=(value,))) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=(record[value_name],)))
        elif value_type == 'vector(int)' or dataset.vocabulary(value_type=value_type) is not None:
            if alts:
                feature_lists[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=value)) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=record[value_name]))
        elif value_type == 'vector(float)':
            if alts:
                feature_lists[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value)) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=record[value_name]))
        elif value_type == 'world':
            features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=record[value_name].flatten()))
    record = tf.train.SequenceExample(context=tf.train.Features(feature=features), feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
    serialized_record = record.SerializeToString()
    return serialized_record


def write_records(dataset, records, path):
    num_records = len(next(iter(records.values())))
    with tf.python_io.TFRecordWriter(path=(path + '.tfrecords.gz'), options=options) as writer:
        for n in range(num_records):
            record = {value_name: value[n] for value_name, value in records.items()}
            serialized_record = write_record(dataset=dataset, record=record)
            writer.write(record=serialized_record)
