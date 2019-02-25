from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters, cnn_size, cnn_depth, cnn_block_depth, world_reduction, embedding_size, rnn, caption_reduction, multimodal_reduction, mlp_size, mlp_depth, soft):

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    if caption_reduction == 'state':
        rnn_size = None
        rnn_state_size = cnn_sizes[-1]
    else:
        rnn_size = cnn_sizes[-1]
        rnn_state_size = None
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=dataset_parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths) >>
        Reduction(reduction=world_reduction, axis=(1, 2))
    )

    caption = (
        (
            Input(name='question', shape=dataset_parameters['question_shape'], dtype='int', tensor=inputs.get('question')) >>
            Embedding(indices=dataset_parameters['vocabulary_size'], size=embedding_size),
            Input(name='question_length', shape=(), dtype='int', tensor=inputs.get('question_length'))
        ) >>
        Rnn(size=rnn_size, state_size=rnn_state_size, cell=rnn)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1)
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    answer = (
        (world, caption) >>
        Reduction(reduction=multimodal_reduction) >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Classification(name='answer', num_classes=dataset_parameters['num_answers'], soft=soft, tensor=inputs.get('answer'))
    )

    return answer
