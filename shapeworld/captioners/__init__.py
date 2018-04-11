import os
from shapeworld import util

directory = os.path.dirname(os.path.realpath(__file__))

if util.v2() and os.path.isfile(os.path.join(directory, 'logical_predication_v2.py')):
    from shapeworld.captioners.logical_predication_v2 import LogicalPredication
else:
    from shapeworld.captioners.logical_predication import LogicalPredication

if util.v2() and os.path.isfile(os.path.join(directory, 'pragmatical_predication_v2.py')):
    from shapeworld.captioners.pragmatical_predication_v2 import PragmaticalPredication
else:
    from shapeworld.captioners.pragmatical_predication import PragmaticalPredication

if util.v2() and os.path.isfile(os.path.join(directory, 'captioner_v2.py')):
    from shapeworld.captioners.captioner_v2 import WorldCaptioner, CaptionerMixer
else:
    from shapeworld.captioners.captioner import WorldCaptioner, CaptionerMixer

if util.v2() and os.path.isfile(os.path.join(directory, 'regular_attribute_v2.py')):
    from shapeworld.captioners.regular_attribute_v2 import RegularAttributeCaptioner
else:
    from shapeworld.captioners.regular_attribute import RegularAttributeCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'regular_type_v2.py')):
    from shapeworld.captioners.regular_type_v2 import RegularTypeCaptioner
else:
    from shapeworld.captioners.regular_type import RegularTypeCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'attribute_type_relation_v2.py')):
    from shapeworld.captioners.attribute_type_relation_v2 import AttributeTypeRelationCaptioner
else:
    from shapeworld.captioners.attribute_type_relation import AttributeTypeRelationCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'relation_v2.py')):
    from shapeworld.captioners.relation_v2 import RelationCaptioner
else:
    from shapeworld.captioners.relation import RelationCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'existential_v2.py')):
    from shapeworld.captioners.existential_v2 import ExistentialCaptioner
else:
    from shapeworld.captioners.existential import ExistentialCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'quantifier_v2.py')):
    from shapeworld.captioners.quantifier_v2 import QuantifierCaptioner
else:
    from shapeworld.captioners.quantifier import QuantifierCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'number_bound_v2.py')):
    from shapeworld.captioners.number_bound_v2 import NumberBoundCaptioner
else:
    from shapeworld.captioners.number_bound import NumberBoundCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'comparative_quantifier_v2.py')):
    from shapeworld.captioners.comparative_quantifier_v2 import ComparativeQuantifierCaptioner
else:
    from shapeworld.captioners.comparative_quantifier import ComparativeQuantifierCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'conjunction_v2.py')):
    from shapeworld.captioners.conjunction_v2 import ConjunctionCaptioner
else:
    from shapeworld.captioners.conjunction import ConjunctionCaptioner

if util.v2() and os.path.isfile(os.path.join(directory, 'disjunction_v2.py')):
    from shapeworld.captioners.disjunction_v2 import DisjunctionCaptioner
else:
    from shapeworld.captioners.disjunction import DisjunctionCaptioner


__all__ = ['LogicalPredication', 'PragmaticalPredication', 'WorldCaptioner', 'CaptionerMixer', 'RegularAttributeCaptioner', 'RegularTypeCaptioner', 'AttributeTypeRelationCaptioner', 'RelationCaptioner', 'ExistentialCaptioner', 'QuantifierCaptioner', 'NumberBoundCaptioner', 'ComparativeQuantifierCaptioner', 'ConjunctionCaptioner', 'DisjunctionCaptioner']
