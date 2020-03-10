import tensorflow as tf
from shapeworld import util
from shapeworld.dataset import LoadedDataset


options = tf.io.TFRecordOptions()  # compression_type='GZIP'


def read_record(dataset, serialized_record):
    features = dict()
    feature_lists = dict()
    for value_name, value_type in dataset.values.items():
        value_type, alts = util.alternatives_type(value_type=value_type)
        if value_type == 'int':
            if alts:
                feature_lists[value_name] = tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64)
            else:
                features[value_name] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        elif value_type == 'float':
            if alts:
                feature_lists[value_name] = tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
            else:
                features[value_name] = tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
        elif value_type == 'vector(int)' or value_type in dataset.vocabularies:
            if alts:
                feature_lists[value_name] = tf.io.FixedLenSequenceFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.int64)
            else:
                features[value_name] = tf.io.FixedLenFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.int64)
        elif value_type == 'vector(float)':
            if alts:
                feature_lists[value_name] = tf.io.FixedLenSequenceFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.float32)
            else:
                features[value_name] = tf.io.FixedLenFeature(shape=dataset.vector_shape(value_name=value_name), dtype=tf.float32)
        elif value_type == 'world':
            if alts:
                feature_lists[value_name] = tf.io.FixedLenSequenceFeature(shape=dataset.world_shape(), dtype=tf.float32)
            else:
                features[value_name] = tf.io.FixedLenFeature(shape=dataset.world_shape(), dtype=tf.float32)
        else:
            pass
    record, sequence_record = tf.io.parse_single_sequence_example(serialized=serialized_record, context_features=features, sequence_features=feature_lists)
    return record, sequence_record


def read_records(dataset, mode):
    paths = tuple(dataset.get_records_paths(mode=mode))
    shuffle = not isinstance(dataset, LoadedDataset) or dataset.random_sampling
    path_queue = tf.train.string_input_producer(string_tensor=paths, shuffle=shuffle, capacity=len(paths))
    reader = tf.TFRecordReader(options=options)
    _, serialized_records = reader.read(queue=path_queue)
    records, sequence_records = read_record(dataset=dataset, serialized_record=serialized_records)
    return records, sequence_records


def batch_records(dataset, mode, batch_size):
    """
    implicit include_model=False
    implicit alternatives=False

    queue runners need to be initialized:

        with tf.Session() as session:
            coordinator = tf.train.Coordinator()
            queue_threads = tf.train.start_queue_runners(sess=session, coord=coordinator)

            # session calls, for instance:
            batch = session.run(fetches=generated)

            coordinator.request_stop()
            coordinator.join(threads=queue_threads)
    """

    with tf.variable_scope(name_or_scope='tf-records'):
        records, sequence_records = read_records(dataset=dataset, mode=mode)
        if not isinstance(dataset, LoadedDataset) or dataset.random_sampling:
            if 'alternatives' in records:
                sample = tf.random_uniform(shape=(), maxval=records['alternatives'], dtype=tf.int64)
                for value_name, sequence_record in sequence_records.items():
                    records[value_name] = sequence_record[sample]
                records.pop('alternatives')
            batch = tf.train.shuffle_batch(tensors=records, batch_size=batch_size, capacity=(batch_size * 50), min_after_dequeue=(batch_size * 10), num_threads=1)
        else:
            if 'alternatives' in records:
                for value_name, sequence_record in sequence_records.items():
                    records[value_name] = sequence_record[0]
                records.pop('alternatives')
            batch = tf.train.batch(tensors=records, batch_size=batch_size, num_threads=1, capacity=(batch_size * 50))
        for value_name in batch:
            value_type, _ = util.alternatives_type(value_type=dataset.values[value_name])
            if dataset.pixel_noise_stddev is not None and dataset.pixel_noise_stddev > 0.0 and value_type == 'world':
                noise = tf.truncated_normal(shape=((batch_size,) + dataset.world_shape()), mean=0.0, stddev=dataset.pixel_noise_stddev)
                batch[value_name] = tf.clip_by_value(t=(batch[value_name] + noise), clip_value_min=0.0, clip_value_max=1.0)
            elif value_type == 'int' or value_type == 'vector(int)' or value_type in dataset.vocabularies:
                batch[value_name] = tf.cast(x=batch[value_name], dtype=tf.int32)
        return batch


def write_record(dataset, record):
    features = dict()
    feature_lists = dict()
    for value_name, value_type in dataset.values.items():
        value_type, alts = util.alternatives_type(value_type=value_type)
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
        elif value_type == 'vector(int)' or value_type in dataset.vocabularies:
            if alts:
                feature_lists[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=value)) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(int64_list=tf.train.Int64List(value=record[value_name]))
        elif value_type == 'vector(float)':
            if alts:
                feature_lists[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten())) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=record[value_name].flatten()))
        elif value_type == 'world':
            if alts:
                features[value_name] = tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten())) for value in record[value_name]])
            else:
                features[value_name] = tf.train.Feature(float_list=tf.train.FloatList(value=record[value_name].flatten()))
    record = tf.train.SequenceExample(context=tf.train.Features(feature=features), feature_lists=tf.train.FeatureLists(feature_list=feature_lists))
    serialized_record = record.SerializeToString()
    return serialized_record


def write_records(dataset, records, path):
    num_records = len(next(iter(records.values())))
    path = str(path)
    if not path.endswith('.tfrecords'):
        path += '.tfrecords'
    with tf.io.TFRecordWriter(path=path, options=options) as writer:
        for n in range(num_records):
            record = {value_name: value[n] for value_name, value in records.items()}
            serialized_record = write_record(dataset=dataset, record=record)
            writer.write(record=serialized_record)
