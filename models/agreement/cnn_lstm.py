from models.TFMacros.tf_macros import *


def model(inputs, **parameters):
    world = (
        Input(name='world', shape=parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=(16, 32, 64), depths=(3, 3, 3)) >>
        Reduction(reduction=parameters.get('world_reduction', 'mean'), axis=(1, 2))
    )
    caption = (
        (
            Input(name='caption', shape=parameters['caption_shape'], dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=parameters['vocabulary_size'], size=32),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=64, unit=Lstm) >>
        Select(index=0) >>
        Reduction(reduction=parameters.get('caption_reduction', 'mean'), axis=1)
    )
    agreement = (
        (world, caption) >>
        Reduction(reduction=parameters.get('multimodal_reduction', 'prod')) >>
        Dense(size=512) >>
        Binary(name='agreement', soft=parameters.get('soft', 0.0), tensor=inputs.get('agreement'))
    )
    return agreement
