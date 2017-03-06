from random import choice


class Texture(object):
    __slots__ = ()

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError

    def model(self):
        return {'name': str(self)}

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        return isinstance(other, Texture) and str(self) == str(other)

    def get_color(self, color, offset):
        raise NotImplementedError

    @staticmethod
    def random_instance(textures, colors, shade_range):
        return choice([Texture.textures[texture] for texture in textures]).random_instance(colors, shade_range)


class SolidTexture(Texture):
    __slots__ = ()

    def __init__(self):
        return super().__init__()

    def __str__(self):
        return 'solid'

    def get_color(self, color, offset):
        return color.get_color()

    @staticmethod
    def random_instance(colors, shade_range):
        return SolidTexture()


Texture.textures = {
    'solid': SolidTexture}
