from models.TFMacros.tf_macros import *


def model(model, inputs, world_shape, resnet_size, resnet_depth, resnet_block_depth, world_reduction, caption_shape, vocabulary_size, embedding_size, caption_reduction, multimodal_reduction, mlp_size, mlp_depth, soft):

    resnet_sizes = [resnet_size * 2**n for n in range(resnet_depth)]
    resnet_depths = [resnet_block_depth for _ in range(resnet_depth)]
    if caption_reduction == 'state':
        assert resnet_sizes[-1] % 2 == 0
        lstm_size = resnet_sizes[-1] // 2
    else:
        lstm_size = resnet_sizes[-1]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=world_shape, tensor=inputs.get('world')) >>
        ResidualNet(sizes=resnet_sizes, depths=resnet_depths) >>
        Reduction(reduction=world_reduction, axis=(1, 2))
    )

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
        (world, caption) >>
        Reduction(reduction=multimodal_reduction) >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
