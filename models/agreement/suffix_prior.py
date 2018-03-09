from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters):

    caption = Input(name='caption', shape=dataset_parameters['caption_shape'], dtype='int', tensor=inputs.get('caption'))()
    caption_length = Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
    agreement = Input(name='agreement', shape=(), dtype='float', tensor=inputs.get('agreement'))()

    agreement = (
        (caption, caption_length, agreement) >>
        SuffixPrior(suffix_length=2, vocabulary_size=dataset_parameters['vocabulary_size']) >>
        Binary(name='agreement', binary_transform=False, tensor=agreement)
    )

    return agreement


class SuffixPrior(Unit):

    num_in = 3
    num_out = 1

    def __init__(self, suffix_length, vocabulary_size):
        super(SuffixPrior, self).__init__()
        self.suffix_length = suffix_length
        self.vocabulary_size = vocabulary_size

    def initialize(self, caption, caption_length, agreement):
        super(SuffixPrior, self).initialize(caption, caption_length, agreement)
        shape = tuple(self.vocabulary_size for _ in range(self.suffix_length)) + (2,)
        self.suffix_agreement_counts = tf.get_variable(name='suffix-agreement-counts', shape=shape, dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32), trainable=False)

    def forward(self, caption, caption_length, agreement):
        super(SuffixPrior, self).forward(caption, caption_length, agreement)
        batch_size = tf.shape(input=caption)[0]
        # -1 because of full stop at the end of a caption
        slice_indices = [tf.stack(values=(tf.range(batch_size), caption_length - 1 - (self.suffix_length - n)), axis=1) for n in range(self.suffix_length)]
        suffix = tf.stack(values=[tf.gather_nd(params=caption, indices=indices) for indices in slice_indices], axis=1)
        agreement_counts = tf.gather_nd(params=self.suffix_agreement_counts, indices=suffix)
        prior = tf.where(
            condition=(agreement_counts[:, 0] > agreement_counts[:, 1]),
            x=tf.zeros(shape=(batch_size,)),
            y=tf.where(
                condition=(agreement_counts[:, 0] < agreement_counts[:, 1]),
                x=tf.ones(shape=(batch_size,)),
                y=(tf.ones(shape=(batch_size,)) * 0.5)
            )
        )
        agreement = tf.expand_dims(input=tf.cast(x=agreement, dtype=Model.dtype('int')), axis=1)
        indices = tf.concat(values=(suffix, agreement), axis=1)
        updates = tf.ones(shape=(batch_size,), dtype=Model.dtype('int'))
        assert Model.current.optimization is None
        Model.current.optimization = tf.scatter_nd_add(ref=self.suffix_agreement_counts, indices=indices, updates=updates)
        return prior
