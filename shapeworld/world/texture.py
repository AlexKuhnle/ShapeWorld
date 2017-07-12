from random import choice


class Texture(object):

    __slots__ = ()

    def __str__(self):
        raise NotImplementedError

    def model(self):
        return {'name': str(self)}

    @staticmethod
    def from_model(model):
        return Texture.textures[model['name']]()

    def copy(self):
        raise NotImplementedError

    def get_color(self, color, offset):
        raise NotImplementedError

    @staticmethod
    def random_instance(textures, colors, shade_range):
        return choice([Texture.textures[texture] for texture in textures]).random_instance(colors, shade_range)


class SolidTexture(Texture):
    __slots__ = ()

    def __init__(self):
        return super(SolidTexture, self).__init__()

    def __str__(self):
        return 'solid'

    def copy(self):
        return SolidTexture()

    def get_color(self, color, offset):
        return color

    @staticmethod
    def random_instance(colors, shade_range):
        return SolidTexture()


Texture.textures = {
    'solid': SolidTexture
}
