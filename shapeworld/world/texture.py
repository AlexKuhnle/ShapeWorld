from random import choice


class Texture(object):

    __slots__ = ()

    def __eq__(self, other):
        return isinstance(other, Texture) and self.name == other.name

    @property
    def name(self):
        raise NotImplementedError

    def model(self):
        return {'name': str(self.name)}

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

    @property
    def name(self):
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
