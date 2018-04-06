import os
from shapeworld import util

directory = os.path.dirname(os.path.realpath(__file__))

if util.v2() and os.path.isfile(os.path.join(directory, 'settings_v2.py')):
    from shapeworld.captions.settings_v2 import Settings
else:
    from shapeworld.captions.settings import Settings

if util.v2() and os.path.isfile(os.path.join(directory, 'caption_v2.py')):
    from shapeworld.captions.caption_v2 import Caption
else:
    from shapeworld.captions.caption import Caption

if util.v2() and os.path.isfile(os.path.join(directory, 'predicate_v2.py')):
    from shapeworld.captions.predicate_v2 import Predicate
else:
    from shapeworld.captions.predicate import Predicate

if util.v2() and os.path.isfile(os.path.join(directory, 'attribute_v2.py')):
    from shapeworld.captions.attribute_v2 import Attribute
else:
    from shapeworld.captions.attribute import Attribute

if util.v2() and os.path.isfile(os.path.join(directory, 'entity_type_v2')):
    from shapeworld.captions.entity_type_v2 import EntityType
else:
    from shapeworld.captions.entity_type import EntityType

if util.v2() and os.path.isfile(os.path.join(directory, 'relation_v2.py')):
    from shapeworld.captions.relation_v2 import Relation
else:
    from shapeworld.captions.relation import Relation

if util.v2() and os.path.isfile(os.path.join(directory, 'existential_v2.py')):
    from shapeworld.captions.existential_v2 import Existential
else:
    from shapeworld.captions.existential import Existential

if util.v2() and os.path.isfile(os.path.join(directory, 'quantifier_v2.py')):
    from shapeworld.captions.quantifier_v2 import Quantifier
else:
    from shapeworld.captions.quantifier import Quantifier

if util.v2() and os.path.isfile(os.path.join(directory, 'number_bound_v2.py')):
    from shapeworld.captions.number_bound_v2 import NumberBound
else:
    from shapeworld.captions.number_bound import NumberBound

if util.v2() and os.path.isfile(os.path.join(directory, 'comparative_quantifier_v2.py')):
    from shapeworld.captions.comparative_quantifier_v2 import ComparativeQuantifier
else:
    from shapeworld.captions.comparative_quantifier import ComparativeQuantifier

if util.v2() and os.path.isfile(os.path.join(directory, 'proposition_v2.py')):
    from shapeworld.captions.proposition_v2 import Proposition
else:
    from shapeworld.captions.proposition import Proposition


__all__ = ['Settings', 'Caption', 'Predicate', 'Attribute', 'EntityType', 'Relation', 'Existential', 'Quantifier', 'NumberBound', 'ComparativeQuantifier', 'Proposition']
