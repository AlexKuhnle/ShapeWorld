from models.TFMacros.tf_macros import *


def model(model, inputs, world_shape, cnn_size, cnn_depth, cnn_block_depth, world_reduction, caption_shape, vocabulary_size, embedding_size, caption_reduction, multimodal_reduction, mlp_size, mlp_depth, soft):

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    conv_size = cnn_sizes[-1]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=world_shape, tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths) >>
        Reduction(reduction=world_reduction, axis=(1, 2))
    )

    caption = (
        Input(name='caption', shape=caption_shape, dtype='int', tensor=inputs.get('caption')) >>
        Embedding(indices=vocabulary_size, size=embedding_size) >>
        NgramConvolution(size=conv_size) >>
        Reduction(reduction=caption_reduction, axis=1)  # not quite!
    )

    agreement = (
        (world, caption) >>
        Reduction(reduction=multimodal_reduction) >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
