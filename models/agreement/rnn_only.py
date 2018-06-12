from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters, embedding_size, rnn, rnn_size, caption_reduction, mlp_size, mlp_depth, soft):

    if caption_reduction == 'state':
        rnn_state_size = rnn_size
        rnn_size = None
    else:
        rnn_state_size = None
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

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
        caption >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
