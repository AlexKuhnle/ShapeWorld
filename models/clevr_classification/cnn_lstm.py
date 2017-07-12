from models.TFMacros.tf_macros import *


def model(inputs, **parameters):
    world = (
        Input(name='world', shape=parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=(16, 32, 64), depths=(3, 3, 3)) >>
        Reduction(reduction=parameters.get('world_reduction', 'mean'), axis=(1, 2))
    )
    question = (
        (
            Input(name='question', shape=parameters['question_shape'], dtype='int', tensor=inputs.get('question')) >>
            Embedding(indices=parameters['vocabulary_size'], size=32),
            Input(name='question_length', shape=1, dtype='int', tensor=inputs.get('question_length'))
        ) >>
        Rnn(size=64, unit=Lstm) >>
        Select(index=0) >>  # embeddings
        Reduction(reduction=parameters.get('question_reduction', 'mean'), axis=1)
    )
    answer = (
        (world, question) >>
        Reduction(reduction=parameters.get('multimodal_reduction', 'prod')) >>
        Dense(size=512) >>
        Classification(name='answer', num_classes=parameters['num_answers'], multi_class=False, soft=parameters.get('soft', 0.0), tensor=inputs.get('answer'))
    )
    return answer
