from random import choice, gauss
import numpy as np


class Color(object):

    __slots__ = ('name', 'rgb', 'shade', 'shaded_rgb')

    def __init__(self, name, rgb, shade):
        assert isinstance(name, str)
        assert isinstance(rgb, tuple) and len(rgb) == 3 and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in rgb)
        assert isinstance(shade, float) and -1.0 <= shade <= 1.0
        self.name = name
        self.rgb = rgb
        self.shade = shade
        if shade < 0.0:
            shaded_rgb = (rgb[0] + rgb[0] * shade, rgb[1] + rgb[1] * shade, rgb[2] + rgb[2] * shade)
        elif shade > 0.0:
            shaded_rgb = (rgb[0] + (1.0 - rgb[0]) * shade, rgb[1] + (1.0 - rgb[1]) * shade, rgb[2] + (1.0 - rgb[2]) * shade)
        else:
            shaded_rgb = rgb
        if rgb == shaded_rgb:
            self.shade = 0.0
        self.shaded_rgb = np.array(object=shaded_rgb, dtype=np.float32)

    def __eq__(self, other):
        return isinstance(other, Color) and self.name == other.name

    def model(self):
        return {'name': str(self.name), 'rgb': list(self.rgb), 'shade': self.shade}

    @staticmethod
    def from_model(model):
        return Color(name=model['name'], rgb=Color.colors[model['name']], shade=model['shade'])

    def copy(self):
        return Color(name=self.name, rgb=self.rgb, shade=self.shade)

    def get_color(self):
        return self.shaded_rgb.copy()

    @staticmethod
    def random_instance(colors, shade_range):
        name, rgb = choice([(color, Color.colors[color]) for color in colors])
        if shade_range > 0.0:
            shade = gauss(mu=0.0, sigma=shade_range)
            while shade < -shade_range or shade > shade_range:
                shade = gauss(mu=0.0, sigma=shade_range)
        else:
            shade = 0.0
        return Color(name, rgb, shade)


Color.colors = {
    'black': (0.0, 0.0, 0.0),
    'red': (1.0, 0.0, 0.0),
    'green': (0.0, 1.0, 0.0),
    'blue': (0.0, 0.0, 1.0),
    'yellow': (1.0, 1.0, 0.0),
    'magenta': (1.0, 0.0, 1.0),
    'cyan': (0.0, 1.0, 1.0),
    'gray': (0.5, 0.5, 0.5)
}
