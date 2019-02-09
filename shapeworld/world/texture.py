from random import choice


class Texture(object):

    __slots__ = ()

    def __eq__(self, other):
        return (isinstance(other, Texture) and self.name == other.name) or (isinstance(other, str) and self.name == other)

    @property
    def name(self):
        return self.__class__.__name__.lower()

    def model(self):
        return dict(name=self.name)

    @staticmethod
    def from_model(model):
        return Texture.textures[model['name']]()

    def copy(self):
        raise NotImplementedError

    def get_color(self, color, offset):
        raise NotImplementedError

    @staticmethod
    def get_textures():
        return sorted(Texture.textures.keys())

    @staticmethod
    def get_texture(name):
        return Texture.textures[name]

    @staticmethod
    def random_instance(colors, shade_range, texture=None, textures=None):
        if texture is not None:
            texture = Texture.get_texture(texture)
        elif textures is not None:
            texture = choice([Texture.get_texture(texture) for texture in textures])
        else:
            assert False
        return texture.random_instance(colors, shade_range)


class Solid(Texture):
    __slots__ = ()

    def __init__(self):
        return super(Solid, self).__init__()

    def copy(self):
        return Solid()

    def get_color(self, color, offset):
        return color

    @staticmethod
    def random_instance(colors, shade_range):
        return Solid()


Texture.textures = dict(
    solid=Solid
)
