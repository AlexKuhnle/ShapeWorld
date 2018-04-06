import os
from shapeworld import util

directory = os.path.dirname(os.path.realpath(__file__))

if util.v2() and os.path.isfile(os.path.join(directory, 'dmrs_v2.py')):
    from shapeworld.realizers.dmrs.dmrs_v2 import Dmrs
else:
    from shapeworld.realizers.dmrs.dmrs import Dmrs

if util.v2() and os.path.isfile(os.path.join(directory, 'realizer_v2.py')):
    from shapeworld.realizers.dmrs.realizer_v2 import DmrsRealizer
else:
    from shapeworld.realizers.dmrs.realizer import DmrsRealizer


__all__ = ['Dmrs', 'DmrsRealizer']
