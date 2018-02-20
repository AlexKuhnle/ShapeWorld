from models.TFMacros.tf_macros import *


def model(model, inputs, **parameters):
    world = (
        Input(name='world', shape=parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=parameters['cnn_sizes'], depths=parameters['cnn_depths']) >>
        Reduction(reduction='collapse', axis=(1, 2))
    )
    caption = (
        (
            Input(name='caption', shape=parameters['caption_shape'], dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=parameters['vocabulary_size'], size=parameters['embedding_size']),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=parameters['lstm_size'], unit=Lstm)
    )
    if parameters['caption_reduction'] == 'state':
        caption >>= Select(index=1) >> Reduction(reduction='concat')
    else:
        caption >>= Select(index=0) >> Reduction(reduction=parameters['caption_reduction'], axis=1)
    agreement = (
        (world, caption) >>
        Repeat(layer=StackedAttention, sizes=parameters['san_sizes']) >>
        Select(index=1) >>
        Repeat(layer=Dense, sizes=parameters['mlp_sizes'], dropout=True) >>
        Binary(name='agreement', soft=parameters['soft'], tensor=inputs.get('agreement'))
    )
    return agreement


class StackedAttention(Layer):

    def __init__(self, size, name=None):
        self.x_linear = Linear(size=size, bias=False)
        self.y_linear = Linear(size=size, bias=True)
        self.activation = Activation(activation='tanh')
        self.dense = Dense(size=0, bias=True, activation='softmax')
        self.reduction = Reduction(reduction='sum', axis=1)
        super(StackedAttention, self).__init__(size=size, name=name)

    def forward(self, x, y):
        attention = ((x >> self.x_linear) + tf.expand_dims(input=(y >> self.y_linear), axis=1)) >> self.activation >> self.dense
        attention = (x * tf.expand_dims(input=attention, axis=2)) >> self.reduction
        return x, y + attention
