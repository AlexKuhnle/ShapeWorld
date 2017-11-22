from models.TFMacros.tf_macros import *


def model(model, inputs, num_classes, multi_class, class_count, world_shape, cnn_size, cnn_depth, cnn_block_depth, world_reduction, mlp_size, mlp_depth, soft):
    assert not class_count

    cnn_sizes = [cnn_size * 2**n for n in range(cnn_depth)]
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    classification = (
        Input(name='world', shape=world_shape, tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths) >>
        Reduction(reduction=world_reduction, axis=(1, 2)) >>
        Repeat(layer=Dense, sizes=mlp_sizes, dropout=True) >>
        Classification(name='classification', num_classes=num_classes, multi_class=multi_class, soft=soft, tensor=inputs.get('classification'))
    )

    return classification
