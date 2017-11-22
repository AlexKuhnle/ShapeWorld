from models.TFMacros.tf_macros import *


def model(model, inputs, num_answers, world_shape, cnn_size, cnn_depth, cnn_block_depth, world_reduction, question_shape, vocabulary_size, embedding_size, caption_reduction, multimodal_reduction, mlp_size, mlp_depth, soft):

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    if caption_reduction == 'state':
        assert cnn_sizes[-1] % 2 == 0
        lstm_size = cnn_sizes[-1] // 2
    else:
        lstm_size = cnn_sizes[-1]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=world_shape, tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths) >>
        Reduction(reduction=world_reduction, axis=(1, 2))
    )

    question = (
        (
            Input(name='question', shape=question_shape, dtype='int', tensor=inputs.get('question')) >>
            Embedding(indices=vocabulary_size, size=embedding_size),
            Input(name='question_length', shape=(), dtype='int', tensor=inputs.get('question_length'))
        ) >>
        Rnn(size=lstm_size, unit=Lstm)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1) >> Reduction(reduction='concat')
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    answer = (
        (world, question) >>
        Reduction(reduction=multimodal_reduction) >>
        Repeat(layer=Dense, sizes=mlp_sizes) >>
        Classification(name='answer', num_classes=num_answers, multi_class=False, soft=soft, tensor=inputs.get('answer'))
    )

    return answer
