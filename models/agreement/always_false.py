from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters):

    agreement = (
        Input(name='world', shape=dataset_parameters['world_shape'], tensor=inputs.get('world')) >>
        Constant(value=False, dtype='float') >>
        Binary(name='agreement', binary_transform=False, tensor=inputs.get('agreement'))
    )

    return agreement
