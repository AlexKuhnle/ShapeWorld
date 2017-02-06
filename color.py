from collections import namedtuple
from random import choice, gauss
import numpy as np


ColorTuple = namedtuple('ColorTuple', ('name', 'rgb', 'shade'))
SolidFillTuple = namedtuple('SolidFillTuple', ('color',))


class Color(ColorTuple):
    __slots__ = ()

    def __new__(cls, name, rgb, shade):
        assert isinstance(name, str)
        assert isinstance(rgb, tuple) and len(rgb) == 3 and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in rgb)
        assert isinstance(shade, float) and -1.0 <= shade <= 1.0
        if shade < 0.0:
            rgb_shade = (rgb[0] + rgb[0] * shade, rgb[1] + rgb[1] * shade, rgb[2] + rgb[2] * shade)
        elif shade > 0.0:
            rgb_shade = (rgb[0] + (1.0 - rgb[0]) * shade, rgb[1] + (1.0 - rgb[1]) * shade, rgb[2] + (1.0 - rgb[2]) * shade)
        else:
            rgb_shade = rgb
        if rgb == rgb_shade:
            shade = 0.0
        return ColorTuple.__new__(cls, name, np.asarray(rgb_shade, dtype=np.float32), shade)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if type(other) is str:
            return str(self) == other
        return self == other

    def __call__(self):
        return np.copy(self.rgb)

    @staticmethod
    def random_instance(colors, shade_range):
        name, rgb = choice(colors)
        if shade_range > 0.0:
            shade = gauss(mu=0.0, sigma=shade_range)
            while shade < -shade_range or shade > shade_range:
                shade = gauss(mu=0.0, sigma=shade_range)
        else:
            shade = 0.0
        return Color(name, rgb, shade)


class Fill(object):
    __slots__ = ()

    def __str__(self):
        raise NotImplementedError

    def __call__(self, offset):
        raise NotImplementedError

    @staticmethod
    def random_instance(fills, colors, shade_range):
        return choice(fills).random_instance(colors, shade_range)


class SolidFill(Fill, SolidFillTuple):
    __slots__ = ()

    def __new__(cls, color):
        assert isinstance(color, Color)
        return SolidFillTuple.__new__(cls, color)

    def __str__(self):
        return 'solid'

    def __call__(self, offset):
        return self.color()

    @staticmethod
    def random_instance(colors, shade_range):
        return SolidFill(Color.random_instance(colors, shade_range))


Fill.fills = {
    'solid': SolidFill}


Color.colors = {
    'black': ('black', (0.0, 0.0, 0.0)),
    'red': ('red', (1.0, 0.0, 0.0)),
    'green': ('green', (0.0, 1.0, 0.0)),
    'blue': ('blue', (0.0, 0.0, 1.0)),
    'yellow': ('yellow', (1.0, 1.0, 0.0)),
    'magenta': ('magenta', (1.0, 0.0, 1.0)),
    'cyan': ('cyan', (0.0, 1.0, 1.0)),
    'white': ('white', (1.0, 1.0, 1.0))}
