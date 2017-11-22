from models.TFMacros.tf_macros import *


def model(model, inputs, world_shape, caption_shape, vocabulary_size, embedding_size, lstm_size, caption_reduction, mlp_size, mlp_depth, soft):

    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    caption = (
        (
            Input(name='caption', shape=caption_shape, dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=vocabulary_size, size=embedding_size),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=lstm_size, unit=Lstm)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1) >> Reduction(reduction='concat')
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    agreement = (
        caption >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
