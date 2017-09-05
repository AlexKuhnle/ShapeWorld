from models.TFMacros.tf_macros import *


def model(model, inputs, **parameters):
    agreement = (
        Input(name='world', shape=parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=(16, 32, 64), depths=(3, 3, 3)) >>
        Reduction(reduction=parameters.get('world_reduction', 'mean'), axis=(1, 2)) >>
        Dense(size=512) >>
        Binary(name='agreement', soft=parameters.get('soft', 0.0), tensor=inputs.get('agreement'))
    )
    return agreement
