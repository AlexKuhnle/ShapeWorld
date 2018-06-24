from models.TFMacros.tf_macros import *


def model(model, inputs, dataset_parameters, cnn_size, cnn_depth, cnn_block_depth, embedding_size, rnn, rnn_size, caption_reduction, film_size, film_depth, conv_size, world_reduction, mlp_size, mlp_depth, soft):

    cnn_sizes = [cnn_size for n in range(cnn_depth)]  # * 2**n
    cnn_depths = [cnn_block_depth for _ in range(cnn_depth)]
    if caption_reduction == 'state':
        rnn_state_size = rnn_size
        rnn_size = None
    else:
        rnn_state_size = None
    film_sizes = [film_size for _ in range(film_depth)]
    mlp_sizes = [mlp_size for _ in range(mlp_depth)]

    world = (
        Input(name='world', shape=dataset_parameters['world_shape'], tensor=inputs.get('world')) >>
        ConvolutionalNet(sizes=cnn_sizes, depths=cnn_depths, final_pool=True)
    )

    caption = (
        (
            Input(name='caption', shape=dataset_parameters['caption_shape'], dtype='int', tensor=inputs.get('caption')) >>
            Embedding(indices=dataset_parameters['vocabulary_size'], size=embedding_size),
            Input(name='caption_length', shape=(), dtype='int', tensor=inputs.get('caption_length'))
        ) >>
        Rnn(size=rnn_size, state_size=rnn_state_size, cell=rnn)
    )

    if caption_reduction == 'state':
        caption >>= Select(index=1)
    else:
        caption >>= Select(index=0) >> Reduction(reduction=caption_reduction, axis=1)

    class CustomFilmLayer(Layer):

        def initialize(self, x):
            super(CustomFilmLayer, self).initialize(x)
            self.conv = Convolution(size=self.size, index=True, window=(1, 1), normalization=False)
            self.film = FiLM(layer=Convolution, size=self.size)

        def forward(self, x):
            super(CustomFilmLayer, self).forward(x)
            x >>= self.conv
            return x + ((x, caption) >> self.film)

    agreement = (
        world >>
        # Convolution(size=film_sizes[0], index=True, window=(3, 3)) >>
        # customize(unit_=Residual, unit=CustomFilmLayer, depth=1)
        Repeat(layer=CustomFilmLayer, sizes=film_sizes) >>
        Convolution(size=conv_size, index=True, window=(1, 1)) >>
        Reduction(reduction=world_reduction, axis=(1, 2)) >>
        Repeat(layer=Dense, sizes=mlp_sizes, activation='relu') >>
        Binary(name='agreement', soft=soft, tensor=inputs.get('agreement'))
    )

    return agreement
