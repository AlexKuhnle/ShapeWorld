from models.TFMacros.tf_macros import *


def model(inputs, **parameters):
    assert not parameters.get('class_count', False)
    classification = (
        Input(name='world', shape=parameters['world_shape'], tensor=inputs.get('world')) >>
        ResidualNet(sizes=(16, 32, 64), depths=(3, 3, 3)) >>
        Reduction(reduction=parameters.get('world_reduction', 'mean'), axis=(1, 2)) >>
        Dense(size=512) >>
        Classification(name='classification', num_classes=parameters['num_classes'], multi_class=parameters.get('multi_class', False), soft=parameters.get('soft', 0.0), tensor=inputs.get('classification'))
    )
    return classification
