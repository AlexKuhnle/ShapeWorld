from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters):

    caption = Input(name='caption', shape=dataset_parameters['caption_shape'], dtype='int', tensor=inputs.get('caption'))()
    agreement = Input(name='agreement', shape=(), dtype='float', tensor=inputs.get('agreement'))()

    agreement = (
        (caption, agreement) >>
        PrefixPrior(prefix_length=2, vocabulary_size=dataset_parameters['vocabulary_size']) >>
        Binary(name='agreement', binary_transform=False, tensor=agreement)
    )

    return agreement


class PrefixPrior(Unit):

    num_in = 2
    num_out = 1

    def __init__(self, prefix_length, vocabulary_size):
        super(PrefixPrior, self).__init__()
        self.prefix_length = prefix_length
        self.vocabulary_size = vocabulary_size

    def initialize(self, caption, agreement):
        super(PrefixPrior, self).initialize(caption, agreement)
        shape = tuple(self.vocabulary_size for _ in range(self.prefix_length)) + (2,)
        self.prefix_agreement_counts = tf.get_variable(name='prefix-agreement-counts', shape=shape, dtype=tf.int32, initializer=tf.zeros_initializer(dtype=tf.int32), trainable=False)

    def forward(self, caption, agreement):
        super(PrefixPrior, self).forward(caption, agreement)
        batch_size = tf.shape(input=caption)[0]
        prefix = caption[:, :self.prefix_length]
        agreement_counts = tf.gather_nd(params=self.prefix_agreement_counts, indices=prefix)
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
        indices = tf.concat(values=(prefix, agreement), axis=1)
        updates = tf.ones(shape=(batch_size,), dtype=Model.dtype('int'))
        assert Model.current.optimization is None
        Model.current.optimization = tf.scatter_nd_add(ref=self.prefix_agreement_counts, indices=indices, updates=updates)
        return prior
