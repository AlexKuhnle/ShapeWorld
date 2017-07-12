from shapeworld.world.shape import Shape
from shapeworld.world.color import Color
from shapeworld.world.texture import Texture
from shapeworld.world.entity import Entity
from shapeworld.world.world import World


all_shapes = Shape.shapes
all_colors = Color.colors
all_textures = Texture.textures


__all__ = ['World', 'Entity', 'Shape', 'Color', 'Texture', 'all_shapes', 'all_colors', 'all_textures']
