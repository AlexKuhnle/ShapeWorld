from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters, cnn_size, cnn_depth, cnn_block_depth, embedding_size, rnn, caption_reduction, san_size, san_depth, mlp_size, mlp_depth, soft):

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    if caption_reduction == 'state':
        rnn_size = None
        rnn_state_size = cnn_sizes[-1]
    else:
        rnn_size = cnn_sizes[-1]
        rnn_state_size = None
    san_sizes = [san_size for _ in range(san_depth)]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=dataset_parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths) >>
        Reduction(reduction='collapse', axis=(1, 2))
    )

    caption = (
        (
            Input(name='caption', shape=dataset_parameters['caption_shape'], dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=dataset_parameters['vocabulary_size'], size=embedding_size),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=rnn_size, state_size=rnn_state_size, cell=rnn)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1)
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    agreement = (
        (world, caption) >>
        Repeat(layer=StackedAttention, sizes=san_sizes) >>
        Select(index=1) >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement


class StackedAttention(Layer):

    num_in = 2
    num_out = 2

    def initialize(self, x, y):
        super(StackedAttention, self).initialize(x)
        self.x_linear = Linear(size=self.size, bias=False)
        self.y_linear = Linear(size=self.size, bias=True)
        self.activation = Activation(activation='tanh')
        self.dense = Dense(size=0, bias=True, activation='softmax')
        self.reduction = Reduction(reduction='sum', axis=1)

    def forward(self, x, y):
        super(StackedAttention, self).forward(x, y)
        attention = ((x >> self.x_linear) + tf.expand_dims(input=(y >> self.y_linear), axis=1)) >> self.activation >> self.dense
        attention = (x * tf.expand_dims(input=attention, axis=2)) >> self.reduction
        return x, y + attention
