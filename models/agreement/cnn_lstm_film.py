from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters, cnn_size, cnn_depth, cnn_block_depth, embedding_size, lstm_size, caption_reduction, film_size, film_depth, conv_size, world_reduction, mlp_size, mlp_depth, soft):

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    film_sizes = [film_size for _ in range(film_depth)]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=dataset_parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths)
    )

    caption = (
        (
            Input(name='caption', shape=dataset_parameters['caption_shape'], dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=dataset_parameters['vocabulary_size'], size=embedding_size),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=lstm_size, unit=Lstm)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1) >> Reduction(reduction='concat')
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    def film_layer(size):
        return (
            (Convolution(size=size, index=True, window=(1, 1), normalization=False), caption) >>
            Convolution(size=size, normalization=(Linear, Linear)) >>
            Select(index=0)
        )

    def res_film_layer(size):
        return Residual(size=size, unit=film_layer, depth=1)

    agreement = (
        world >>
        Convolution(size=film_sizes[0], index=True, window=(3, 3), normalization=False) >>
        Repeat(layer=res_film_layer, sizes=film_sizes) >>
        Convolution(size=conv_size, window=(1, 1)) >>
        Reduction(reduction=world_reduction, axis=(1, 2)) >>
        Repeat(layer=Dense, sizes=mlp_sizes) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
