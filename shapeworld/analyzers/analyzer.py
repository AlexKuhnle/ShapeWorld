import copy
import json
import os
import re
from pydmrs.matching.exact_matching import dmrs_exact_matching
from shapeworld.analyzers.ace import Ace
from shapeworld.captions import Attribute, Relation, EntityType, Selector, Existential, Quantifier, NumberBound, ComparativeQuantifier, Proposition
from shapeworld.realizers.dmrs.dmrs import Dmrs, SubDmrs, create_sortinfo
from shapeworld.realizers.dmrs.realizer import prepare_ace, prepare_grammar


int_regex = re.compile(pattern=r'^-?[0-9]+$')
float_regex = re.compile(pattern=r'^-?[0-9]+.[0-9]+$')
nested_tuple_regex = re.compile(pattern=r'^([a-z]+(\+[a-z]+)+)(,([a-z]+(\+[a-z]+)+))+$')
tuple_regex = re.compile(pattern=r'^[a-z]+(,[a-z]+)+$')


def parse_string(string):
    if int_regex.match(string):
        return int(string)
    elif float_regex.match(string):
        return float(string)
    elif nested_tuple_regex.match(string):
        return tuple(tuple(s.split('+')) for s in string.split(','))
    elif tuple_regex.match(string):
        return tuple(string.split(','))
    else:
        return str(string)


class DmrsAnalyzer(object):

    def __init__(self, language):
        prepare_ace()
        prepare_grammar(language=language)
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'realizers', 'dmrs')
        self.ace_path = os.path.join(directory, 'resources', 'ace')
        self.erg_path = os.path.join(directory, 'languages', language + '.dat')

        self.ace = Ace(executable=self.ace_path, grammar=self.erg_path)

        with open(os.path.join(directory, 'languages', language + '.json'), 'r') as filehandle:
            language = json.load(fp=filehandle)

        if 'sortinfos' in language:
            sortinfo_classes = dict()
            sortinfo_shortforms = dict()
            for cvarsort, sortinfo in language['sortinfos'].items():
                assert 'features' in sortinfo
                sortinfo_class = create_sortinfo(cvarsort, tuple(sortinfo['features']))
                sortinfo_classes[cvarsort] = sortinfo_class
                if 'shortform' in sortinfo:
                    shortform = sortinfo['shortform']
                    assert all(feature in sortinfo_class.features for feature in shortform)
                    assert all(len(key) == 1 and key not in '_?' for feature, kvs in shortform.items() for key in kvs)
                    sortinfo_shortforms[cvarsort] = shortform
        else:
            sortinfo_classes = None
            sortinfo_shortforms = None

        self.unused = set()

        self.attributes = dict()
        self.attribute_by_key = dict()
        self.relation_attribute = None
        if 'attributes' in language:
            for predtype, values in language['attributes'].items():
                predtype = parse_string(predtype)
                if predtype == 'relation':
                    self.relation_attribute = Dmrs.parse(values['dmrs'])
                    self.unused.add(('attribute', 'relation'))
                    continue
                elif predtype not in self.attributes:
                    self.attributes[predtype] = dict()
                for value, attribute in values.items():
                    value = parse_string(value)
                    self.attributes[predtype][value] = Dmrs.parse(attribute['dmrs'])
                    assert attribute['key'] not in self.attribute_by_key
                    self.attribute_by_key[attribute['key']] = (predtype, value)
                    self.unused.add(('attribute', predtype, value))

        self.entity_type = None
        if 'type' in language:
            self.entity_type = Dmrs.parse(language['type']['dmrs'])
            self.unused.add(('type',))

        self.selectors = dict()
        self.selector_by_key = dict()
        self.unique_selector = None
        if 'selectors' in language:
            for predtype, values in language['selectors'].items():
                predtype = parse_string(predtype)
                if predtype == 'unique':
                    self.unique_selector = Dmrs.parse(values['dmrs'])
                    self.unused.add(('selector', 'unique'))
                    continue
                elif predtype not in self.selectors:
                    self.selectors[predtype] = dict()
                for value, selector in values.items():
                    value = parse_string(value)
                    self.selectors[predtype][value] = Dmrs.parse(selector['dmrs'])
                    assert selector['key'] not in self.selector_by_key
                    self.selector_by_key[selector['key']] = (predtype, value)
                    self.unused.add(('selector', predtype, value))

        self.relations = dict()
        self.relation_by_key = dict()
        self.attribute_relation = None
        self.type_relation = None
        if 'relations' in language:
            for predtype, values in language['relations'].items():
                predtype = parse_string(predtype)
                if predtype == 'attribute':
                    self.attribute_relation = Dmrs.parse(values['dmrs'])
                    self.unused.add(('relation', 'attribute'))
                    continue
                elif predtype == 'type':
                    self.type_relation = Dmrs.parse(values['dmrs'])
                    self.unused.add(('relation', 'type'))
                    continue
                elif predtype not in self.relations:
                    self.relations[predtype] = dict()
                for value, relation in values.items():
                    value = parse_string(value)
                    self.relations[predtype][value] = Dmrs.parse(relation['dmrs'])
                    assert relation['key'] not in self.relation_by_key
                    self.relation_by_key[relation['key']] = (predtype, value)
                    self.unused.add(('relation', predtype, value))

        self.existential = None
        self.type_existential = None
        self.selector_existential = None
        if 'existential' in language:
            if 'type' in language['existential']:
                self.type_existential = Dmrs.parse(language['existential']['type']['dmrs'])
                self.unused.add(('existential', 'type'))
            if 'selector' in language['existential']:
                self.selector_existential = Dmrs.parse(language['existential']['selector']['dmrs'])
                self.unused.add(('existential', 'selector'))

        self.quantifiers = dict()
        self.quantifier_by_key = dict()
        if 'quantifiers' in language:
            for qtype, qranges in language['quantifiers'].items():
                qtype = parse_string(qtype)
                if qtype not in self.quantifiers:
                    self.quantifiers[qtype] = dict()
                if qtype == 'composed':
                    for identifier, quantifier in qranges.items():
                        identifier = parse_string(identifier)
                        definition = tuple((str(qtype), str(qrange), quantity) for qtype, qrange, quantity in quantifier.pop('definition'))
                        self.quantifiers[qtype][identifier] = {definition: Dmrs.parse(quantifier['dmrs'])}
                        assert identifier not in self.quantifier_by_key
                        self.quantifier_by_key[identifier] = (qtype, identifier, definition)
                        self.unused.add(('quantifier', qtype, identifier, definition))
                    continue
                for qrange, quantities in qranges.items():
                    qrange = parse_string(qrange)
                    if qrange not in self.quantifiers[qtype]:
                        self.quantifiers[qtype][qrange] = dict()
                    for quantity, quantifier in quantities.items():
                        quantity = parse_string(quantity)
                        self.quantifiers[qtype][qrange][quantity] = Dmrs.parse(quantifier['dmrs'])
                        assert quantifier['key'] not in self.quantifier_by_key
                        self.quantifier_by_key[quantifier['key']] = (qtype, qrange, quantity)
                        self.unused.add(('quantifier', qtype, qrange, quantity))

        self.number_bounds = dict()
        self.number_bound_by_key = dict()
        if 'number-bounds' in language:
            for bound, number_bound in language['number-bounds'].items():
                bound = parse_string(bound)
                self.number_bounds[bound] = Dmrs.parse(number_bound['dmrs'])
                assert number_bound['key'] not in self.number_bound_by_key
                self.number_bound_by_key[number_bound['key']] = (bound,)
                self.unused.add(('number-bound', bound))

        self.comparative_quantifiers = dict()
        self.comparative_quantifier_by_key = dict()
        if 'comparative-quantifiers' in language:
            for qtype, qranges in language['comparative-quantifiers'].items():
                qtype = parse_string(qtype)
                if qtype not in self.comparative_quantifiers:
                    self.comparative_quantifiers[qtype] = dict()
                if qtype == 'composed':
                    for identifier, comparative_quantifier in qranges.items():
                        identifier = parse_string(identifier)
                        definition = tuple((str(qtype), str(qrange), quantity) for qtype, qrange, quantity in comparative_quantifier.pop('definition'))
                        self.comparative_quantifiers[qtype][identifier] = {definition: Dmrs.parse(comparative_quantifier['dmrs'])}
                        assert identifier not in self.comparative_quantifier_by_key
                        self.comparative_quantifier_by_key[identifier] = (qtype, identifier, definition)
                        self.unused.add(('comparative-quantifier', qtype, identifier, definition))
                    continue
                for qrange, quantities in qranges.items():
                    qrange = parse_string(qrange)
                    if qrange not in self.comparative_quantifiers[qtype]:
                        self.comparative_quantifiers[qtype][qrange] = dict()
                    for quantity, quantifier in quantities.items():
                        quantity = parse_string(quantity)
                        self.comparative_quantifiers[qtype][qrange][quantity] = Dmrs.parse(quantifier['dmrs'])
                        assert quantifier['key'] not in self.comparative_quantifier_by_key
                        self.comparative_quantifier_by_key[quantifier['key']] = (qtype, qrange, quantity)
                        self.unused.add(('comparative-quantifier', qtype, qrange, quantity))

        self.propositions = dict()
        self.proposition_by_key = dict()
        for connective, proposition in language['propositions'].items():
            connective = parse_string(connective)
            if isinstance(proposition['dmrs'], list):
                self.propositions[connective] = tuple(Dmrs.parse(dmrs) for dmrs in proposition['dmrs'])
            else:
                self.propositions[connective] = Dmrs.parse(proposition['dmrs'])
            assert proposition['key'] not in self.proposition_by_key
            self.proposition_by_key[proposition['key']] = connective
            self.unused.add(('proposition', connective))

        self.hierarchy = language['hierarchy']

        self.pre_processing = list()
        self.pre_processing_by_key = dict()
        for n, paraphrase in enumerate(reversed(language['post-processing'])):
            search = Dmrs.parse(paraphrase['replace'])
            replace = Dmrs.parse(paraphrase.get('reverse', paraphrase['search']))
            disable_hierarchy = paraphrase.get('disable_hierarchy', False)
            match_top_index = paraphrase.get('match_top_index', False)
            self.pre_processing.append((search, replace, disable_hierarchy, match_top_index))
            assert paraphrase['key'] not in self.pre_processing_by_key
            self.pre_processing_by_key[paraphrase['key']] = n

    def analyze(self, sentences):
        captions = list()
        mrs_iter_iter = self.ace.parse(sentence_list=sentences)
        for mrs_iter in mrs_iter_iter:
            for k, mrs in enumerate(mrs_iter):
                if mrs is None:
                    continue
                try:
                    dmrs = mrs.convert_to(cls=Dmrs, copy_nodes=True)
                except Exception:
                    continue
                analyses = self.analyze2(dmrs=dmrs)
                try:
                    captions.append(next(analyses))
                    break
                except StopIteration:
                    continue
            else:
                captions.append(None)
        return captions

    def analyze2(self, dmrs):
        # print(dmrs.dumps_xml())
        for search, replace, disable_hierarchy, match_top_index in self.pre_processing:
            dmrs = dmrs.apply_paraphrases(paraphrases=[(search, replace)], hierarchy=(None if disable_hierarchy else self.hierarchy), match_top_index=match_top_index)
        # if any(str(node.pred) == '_if+and+only+if_x_1' for node in dmrs.iter_nodes()) and all(str(node.pred) != 'generic_entity' for node in dmrs.iter_nodes()):
        # print(dmrs.dumps_xml())

        # if any(str(node.pred) == 'much-many_a' for node in dmrs.iter_nodes()) and all(str(node.pred) != 'generic_entity' for node in dmrs.iter_nodes()) and all(str(node.pred) != 'number_q' for node in dmrs.iter_nodes()) and all(str(node.pred) != 'loc_nonsp' for node in dmrs.iter_nodes()):
        #     print(dmrs.dumps_xml())
        # print('analyse', dmrs.dumps_xml())
        # for caption_type in ('attribute', 'type', 'relation', 'existential', 'quantifier', 'number_bound', 'comparative_quantifier'):
        for caption, caption_dmrs in self.caption_with_dmrs(dmrs=dmrs):
            matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=True))
            assert len(matches) <= 1
            if len(matches) == 1 and len(caption_dmrs) == len(dmrs) and all(dmrs[matches[0][nodeid]].pred == caption_dmrs[nodeid].pred for nodeid in caption_dmrs):
                yield caption

    def attribute_caption(self, dmrs):
        # predtype: relation
        matches = list(dmrs_exact_matching(sub_dmrs=self.relation_attribute, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('attribute > relation')
            relation_nodeid = match[self.relation_attribute.anchors['attr'].nodeid]
            unmatched_dmrs = dmrs.subgraph(nodeid=relation_nodeid)
            for relation, relation_dmrs in self.relation_caption(dmrs=unmatched_dmrs):
                attribute_dmrs = copy.deepcopy(self.relation_attribute)
                attribute_dmrs.compose(relation_dmrs, fusion={'attr': 'rel'}, hierarchy=self.hierarchy)
                matches = list(dmrs_exact_matching(sub_dmrs=attribute_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                if len(matches) >= 1:
                    attribute = Attribute(predtype='relation', value=relation)
                    self.unused.discard(('attribute', 'relation'))
                    yield attribute, attribute_dmrs

        # predtype: *
        for predtype in self.attributes:
            for value in self.attributes[predtype]:
                # print(predtype, value)
                # print([str(node.pred) for node in self.attributes[predtype][value].iter_nodes()])
                # print([str(node.pred) for node in dmrs.iter_nodes()])
                matches = list(dmrs_exact_matching(sub_dmrs=self.attributes[predtype][value], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                for match in matches:
                    # print('attribute >', predtype, value)
                    attribute_dmrs = copy.deepcopy(self.attributes[predtype][value])
                    attribute = Attribute(predtype=predtype, value=value)
                    self.unused.discard(('attribute', predtype, value))
                    yield attribute, attribute_dmrs

    def type_caption(self, dmrs):  # entity_ not if suffix
        matches = list(dmrs_exact_matching(sub_dmrs=self.entity_type, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('type > !')
            type_nodeid = match[self.entity_type.anchors['type'].nodeid]
            quant_nodeid = match[self.entity_type.anchors['quant'].nodeid]

            def next_attribute_caption(type_dmrs, unmatched_dmrs):
                no_match = True
                for attribute, attribute_dmrs in self.attribute_caption(dmrs=unmatched_dmrs):
                    if type_dmrs.anchors['type'].pred != attribute_dmrs.anchors['type'].pred and not (type_dmrs.anchors['type'].pred.is_less_specific(attribute_dmrs.anchors['type'].pred, hierarchy=self.hierarchy) or type_dmrs.anchors['type'].pred.is_more_specific(attribute_dmrs.anchors['type'].pred, hierarchy=self.hierarchy)):
                        continue
                    next_type_dmrs = copy.deepcopy(type_dmrs)
                    next_type_dmrs.compose(attribute_dmrs, fusion={'type': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                    if all(nodeid in type_dmrs and next_type_dmrs[nodeid].pred == type_dmrs[nodeid].pred for nodeid in next_type_dmrs):
                        continue
                    matches = list(dmrs_exact_matching(sub_dmrs=next_type_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    for match in matches:
                        no_match = False
                        next_unmatched_dmrs = copy.deepcopy(unmatched_dmrs)
                        next_unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in next_type_dmrs if match[nodeid] != type_nodeid and match[nodeid] != quant_nodeid)
                        # attributes.append(attribute)
                        for entity_type, next_type_dmrs in list(next_attribute_caption(type_dmrs=next_type_dmrs, unmatched_dmrs=next_unmatched_dmrs)):
                            entity_type.value.append(attribute)
                            yield entity_type, next_type_dmrs
                if no_match:
                    entity_type = EntityType()
                    yield entity_type, type_dmrs

            type_dmrs = copy.deepcopy(self.entity_type)
            unmatched_dmrs = SubDmrs(dmrs=dmrs)
            # attributes = list()
            for entity_type, type_dmrs in list(next_attribute_caption(type_dmrs=type_dmrs, unmatched_dmrs=unmatched_dmrs)):
                self.unused.discard(('type',))
                yield entity_type, type_dmrs

    def selector_caption(self, dmrs):
        # predtype: unique
        matches = list(dmrs_exact_matching(sub_dmrs=self.unique_selector, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('selector > unique')
            unmatched_dmrs = SubDmrs(dmrs=dmrs)
            for entity_type, type_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                selector_dmrs = copy.deepcopy(self.unique_selector)
                selector_dmrs.compose(type_dmrs, fusion={'scope': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                matches = list(dmrs_exact_matching(sub_dmrs=selector_dmrs, dmrs=dmrs, hierarchy=self.hierarchy))
                if len(matches) >= 1:
                    selector = Selector(predtype='unique', scope=entity_type)
                    self.unused.discard(('selector', 'unique'))
                    yield selector, selector_dmrs

        # predtype: *
        for predtype in self.selectors:
            for value in self.selectors[predtype]:
                selector_dmrs = self.selectors[predtype][value]
                matches = list(dmrs_exact_matching(sub_dmrs=selector_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                for match in matches:
                    # print('selector >', predtype, value)
                    selector_nodeid = match[selector_dmrs.anchors['sel'].nodeid]
                    scope_nodeid = match[selector_dmrs.anchors['scope'].nodeid]
                    if predtype in Selector.comparison_selectors:
                        comparison_nodeid = match[selector_dmrs.anchors['comp'].nodeid]
                        unmatched_dmrs = dmrs.subgraph(nodeid=scope_nodeid, exclude=(selector_nodeid, comparison_nodeid))
                    else:
                        unmatched_dmrs = dmrs.subgraph(nodeid=scope_nodeid, exclude=(selector_nodeid,))
                    for scope, scope_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                        # print('selector > scope')
                        scope_selector_dmrs = copy.deepcopy(selector_dmrs)
                        scope_selector_dmrs.compose(scope_dmrs, fusion={'scope': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                        matches = list(dmrs_exact_matching(sub_dmrs=scope_selector_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                        for match in matches:
                            if predtype in Selector.comparison_selectors:
                                unmatched_dmrs = dmrs.subgraph(nodeid=comparison_nodeid, exclude=(selector_nodeid, scope_nodeid))
                                unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in scope_selector_dmrs if match[nodeid] != comparison_nodeid)
                                for comparison, comparison_dmrs in self.selector_caption(dmrs=unmatched_dmrs):
                                    # print('selector > comparison')
                                    comp_selector_dmrs = copy.deepcopy(scope_selector_dmrs)
                                    comp_selector_dmrs.compose(comparison_dmrs, fusion={'comp': 'scope'}, hierarchy=self.hierarchy)
                                    matches = list(dmrs_exact_matching(sub_dmrs=comp_selector_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                    if len(matches) >= 1:
                                        selector = Selector(predtype=predtype, value=value, scope=scope, comparison=comparison)
                                        self.unused.discard(('relation', predtype, value))
                                        yield selector, comp_selector_dmrs
                            else:
                                selector = Selector(predtype=predtype, value=value, scope=scope)
                                self.unused.discard(('selector', predtype, value))
                                yield selector, scope_selector_dmrs

    def relation_caption(self, dmrs):
        # predtype: attribute
        matches = list(dmrs_exact_matching(sub_dmrs=self.attribute_relation, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('relation > attribute')
            relation_nodeid = match[self.attribute_relation.anchors['rel'].nodeid]
            type_nodeid = match[self.attribute_relation.anchors['type'].nodeid]
            unmatched_dmrs = dmrs.subgraph(nodeid=type_nodeid, exclude=(relation_nodeid,))
            for attribute, attribute_dmrs in self.attribute_caption(dmrs=unmatched_dmrs):
                relation_dmrs = copy.deepcopy(self.attribute_relation)
                relation_dmrs.compose(attribute_dmrs, fusion={'type': 'type'}, hierarchy=self.hierarchy)
                matches = list(dmrs_exact_matching(sub_dmrs=relation_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                if len(matches) >= 1:
                    relation = Relation(predtype='attribute', value=attribute)
                    self.unused.discard(('relation', 'attribute'))
                    yield relation, relation_dmrs

        # predtype: type
        matches = list(dmrs_exact_matching(sub_dmrs=self.type_relation, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('relation > type')
            relation_nodeid = match[self.type_relation.anchors['rel'].nodeid]
            type_nodeid = match[self.type_relation.anchors['type'].nodeid]
            unmatched_dmrs = dmrs.subgraph(nodeid=type_nodeid, exclude=(relation_nodeid,))
            for entity_type, type_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                relation_dmrs = copy.deepcopy(self.type_relation)
                relation_dmrs.compose(type_dmrs, fusion={'type': 'type'}, hierarchy=self.hierarchy)
                matches = list(dmrs_exact_matching(sub_dmrs=relation_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                if len(matches) >= 1:
                    relation = Relation(predtype='type', value=entity_type)
                    self.unused.discard(('relation', 'type'))
                    yield relation, relation_dmrs

        # predtype: *
        for predtype in self.relations:
            for value in self.relations[predtype]:
                relation_dmrs = self.relations[predtype][value]
                matches = list(dmrs_exact_matching(sub_dmrs=relation_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                for match in matches:
                    # print('relation >', predtype, value)
                    # print([str(node.pred) for node in self.relations[predtype][value].iter_nodes()])
                    # print([str(node.pred) for node in dmrs.iter_nodes()])
                    relation_nodeid = match[relation_dmrs.anchors['rel'].nodeid]
                    reference_nodeid = match[relation_dmrs.anchors['ref'].nodeid]
                    if predtype in Relation.meta_relations:
                        unmatched_dmrs = dmrs.subgraph(nodeid=reference_nodeid, exclude=(relation_nodeid,))
                        unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in relation_dmrs if match[nodeid] != reference_nodeid)
                    else:
                        quantifier_nodeid = match[relation_dmrs.anchors['quant'].nodeid]
                        if predtype in Relation.ternary_relations:
                            comparison_nodeid = match[relation_dmrs.anchors['comp'].nodeid]
                            unmatched_dmrs = dmrs.subgraph(nodeid=reference_nodeid, exclude=(relation_nodeid, comparison_nodeid))
                        else:
                            unmatched_dmrs = dmrs.subgraph(nodeid=reference_nodeid, exclude=(relation_nodeid,))
                        unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in relation_dmrs if match[nodeid] not in (reference_nodeid, quantifier_nodeid))
                    if predtype in Relation.meta_relations:
                        reference_iter = self.relation_caption(dmrs=unmatched_dmrs)
                    else:
                        reference_iter = self.type_caption(dmrs=unmatched_dmrs)
                    for reference, reference_dmrs in reference_iter:
                        # print('relation > reference')
                        ref_relation_dmrs = copy.deepcopy(relation_dmrs)
                        if predtype in Relation.meta_relations:
                            ref_relation_dmrs.compose(reference_dmrs, fusion={'ref': 'rel'}, hierarchy=self.hierarchy)
                        else:
                            ref_relation_dmrs.compose(reference_dmrs, fusion={'ref': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                        matches = list(dmrs_exact_matching(sub_dmrs=ref_relation_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                        for match in matches:
                            if predtype in Relation.ternary_relations:
                                unmatched_dmrs = dmrs.subgraph(nodeid=comparison_nodeid, exclude=(relation_nodeid, reference_nodeid))
                                unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in ref_relation_dmrs if match[nodeid] != comparison_nodeid)
                                for comparison, comparison_dmrs in self.selector_caption(dmrs=unmatched_dmrs):
                                    # print('relation > comparison')
                                    comp_relation_dmrs = copy.deepcopy(ref_relation_dmrs)
                                    comp_relation_dmrs.compose(comparison_dmrs, fusion={'comp': 'scope'}, hierarchy=self.hierarchy)
                                    matches = list(dmrs_exact_matching(sub_dmrs=comp_relation_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                    if len(matches) >= 1:
                                        relation = Relation(predtype=predtype, value=value, reference=reference, comparison=comparison)
                                        self.unused.discard(('relation', predtype, value))
                                        yield relation, comp_relation_dmrs
                            else:
                                relation = Relation(predtype=predtype, value=value, reference=reference)
                                self.unused.discard(('relation', predtype, value))
                                yield relation, ref_relation_dmrs

    def existential_caption(self, dmrs):
        matches = list(dmrs_exact_matching(sub_dmrs=self.type_existential, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('existential type > restrictor')
            restrictor_nodeid = match[self.type_existential.anchors['rstr'].nodeid]
            body_nodeid = match[self.type_existential.anchors['body'].nodeid]
            unmatched_dmrs = dmrs.subgraph(nodeid=restrictor_nodeid, exclude=(body_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
            for restrictor, restrictor_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                rstr_existential_dmrs = copy.deepcopy(self.type_existential)
                rstr_existential_dmrs.compose(restrictor_dmrs, fusion={'rstr': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                matches = list(dmrs_exact_matching(sub_dmrs=rstr_existential_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                if len(matches) >= 1:
                    # print('existential type > body')
                    for match in matches:
                        unmatched_dmrs = dmrs.subgraph(nodeid=body_nodeid, exclude=(restrictor_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                        unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in rstr_existential_dmrs if match[nodeid] != body_nodeid)
                        for body, body_dmrs in self.relation_caption(dmrs=unmatched_dmrs):
                            body_existential_dmrs = copy.deepcopy(rstr_existential_dmrs)
                            body_existential_dmrs.compose(body_dmrs, fusion={'body': 'rel'}, hierarchy=self.hierarchy)
                            matches = list(dmrs_exact_matching(sub_dmrs=body_existential_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                            if len(matches) >= 1:
                                existential = Existential(restrictor=restrictor, body=body)
                                self.unused.discard(('existential', 'type'))
                                yield existential, body_existential_dmrs

        matches = list(dmrs_exact_matching(sub_dmrs=self.selector_existential, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
        for match in matches:
            # print('existential selector > restrictor')
            restrictor_nodeid = match[self.selector_existential.anchors['rstr'].nodeid]
            body_nodeid = match[self.selector_existential.anchors['body'].nodeid]
            unmatched_dmrs = dmrs.subgraph(nodeid=restrictor_nodeid, exclude=(body_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
            for restrictor, restrictor_dmrs in self.selector_caption(dmrs=unmatched_dmrs):
                rstr_existential_dmrs = copy.deepcopy(self.selector_existential)
                rstr_existential_dmrs.compose(restrictor_dmrs, fusion={'rstr': 'scope'}, hierarchy=self.hierarchy)
                matches = list(dmrs_exact_matching(sub_dmrs=rstr_existential_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                if len(matches) >= 1:
                    # print('existential selector > body')
                    for match in matches:
                        unmatched_dmrs = dmrs.subgraph(nodeid=body_nodeid, exclude=(restrictor_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                        unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in rstr_existential_dmrs if match[nodeid] != body_nodeid)
                        for body, body_dmrs in self.relation_caption(dmrs=unmatched_dmrs):
                            body_existential_dmrs = copy.deepcopy(rstr_existential_dmrs)
                            body_existential_dmrs.compose(body_dmrs, fusion={'body': 'rel'}, hierarchy=self.hierarchy)
                            matches = list(dmrs_exact_matching(sub_dmrs=body_existential_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                            if len(matches) >= 1:
                                existential = Existential(restrictor=restrictor, body=body)
                                self.unused.discard(('existential', 'selector'))
                                yield existential, body_existential_dmrs

    def quantifier_caption(self, dmrs):
        for qtype in self.quantifiers:
            for qrange in self.quantifiers[qtype]:
                for quantity in self.quantifiers[qtype][qrange]:
                    # if any(str(node.pred) == '_at+least_x_deg' for node in dmrs.iter_nodes()) and any(str(node.pred) == '_quarter_n_of' for node in dmrs.iter_nodes()) and any(str(node.pred) == '_at+least_x_deg' for node in self.quantifiers[qtype][qrange][quantity].iter_nodes()):
                    #     print([str(node) for node in dmrs.iter_nodes()])
                    #     print([str(node) for node in self.quantifiers[qtype][qrange][quantity].iter_nodes()])
                    quantifier_dmrs = self.quantifiers[qtype][qrange][quantity]
                    matches = list(dmrs_exact_matching(sub_dmrs=quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    for match in matches:
                        # print('quantifier > restrictor', qtype, qrange, quantity)
                        restrictor_nodeid = match[quantifier_dmrs.anchors['rstr'].nodeid]
                        body_nodeid = match[quantifier_dmrs.anchors['body'].nodeid]
                        unmatched_dmrs = dmrs.subgraph(nodeid=restrictor_nodeid, exclude=(body_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                        for restrictor, restrictor_dmrs in self.type_caption(dmrs=unmatched_dmrs):  # only one?
                            rstr_quantifier_dmrs = copy.deepcopy(quantifier_dmrs)
                            rstr_quantifier_dmrs.compose(restrictor_dmrs, fusion={'rstr': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                            # print('rstr', rstr_quantifier_dmrs.dumps_xml())
                            # print('dmrs', dmrs.dumps_xml())
                            matches = list(dmrs_exact_matching(sub_dmrs=rstr_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                            for match in matches:
                                # print('quantifier > body')
                                unmatched_dmrs = dmrs.subgraph(nodeid=body_nodeid, exclude=(restrictor_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                                unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in rstr_quantifier_dmrs if match[nodeid] != body_nodeid)
                                for body, body_dmrs in self.relation_caption(dmrs=unmatched_dmrs):  # only one?
                                    body_quantifier_dmrs = copy.deepcopy(rstr_quantifier_dmrs)
                                    body_quantifier_dmrs.compose(body_dmrs, fusion={'body': 'rel'}, hierarchy=self.hierarchy)
                                    matches = list(dmrs_exact_matching(sub_dmrs=body_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                    if len(matches) >= 1:
                                        quantifier = Quantifier(qtype=qtype, qrange=qrange, quantity=quantity, restrictor=restrictor, body=body)
                                        self.unused.discard(('quantifier', qtype, qrange, quantity))
                                        yield quantifier, body_quantifier_dmrs

    def number_bound_caption(self, dmrs):
        for qtype in self.quantifiers:
            for qrange in self.quantifiers[qtype]:
                for quantity in self.quantifiers[qtype][qrange]:
                    quantifier_dmrs = self.quantifiers[qtype][qrange][quantity]
                    matches = list(dmrs_exact_matching(sub_dmrs=quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    for match in matches:
                        # print('number-bound > quantifier > restrictor', qtype, qrange, quantity)
                        restrictor_nodeid = match[quantifier_dmrs.anchors['rstr'].nodeid]
                        body_nodeid = match[quantifier_dmrs.anchors['body'].nodeid]
                        unmatched_dmrs = dmrs.subgraph(nodeid=restrictor_nodeid, exclude=(body_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                        for bound in self.number_bounds:
                            bound_dmrs = self.number_bounds[bound]
                            matches = list(dmrs_exact_matching(sub_dmrs=bound_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                            for match in matches:
                                # print('number-bound >', bound)
                                scope_nodeid = match[bound_dmrs.anchors['scope'].nodeid]
                                type_nodeid = match[bound_dmrs.anchors['type'].nodeid]
                                unmatched_dmrs = dmrs.subgraph(nodeid=scope_nodeid, exclude=(type_nodeid, body_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                                for restrictor, restrictor_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                                    rstr_bound_dmrs = copy.deepcopy(bound_dmrs)
                                    rstr_bound_dmrs.compose(restrictor_dmrs, fusion={'scope': 'type', 'tquant': 'quant'}, hierarchy=self.hierarchy)
                                    rstr_quantifier_dmrs = copy.deepcopy(quantifier_dmrs)
                                    rstr_quantifier_dmrs.compose(rstr_bound_dmrs, fusion={'rstr': 'type', 'quant': 'quant'}, hierarchy=self.hierarchy)
                                    matches = list(dmrs_exact_matching(sub_dmrs=rstr_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                    for match in matches:
                                        # print('number-bound > quantifier > body')
                                        unmatched_dmrs = dmrs.subgraph(nodeid=body_nodeid, exclude=(restrictor_nodeid,))  # dmrs.index.nodeid, dmrs.top.nodeid
                                        unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in rstr_quantifier_dmrs if match[nodeid] != body_nodeid)
                                        for body, body_dmrs in self.relation_caption(dmrs=unmatched_dmrs):  # only one?
                                            body_quantifier_dmrs = copy.deepcopy(rstr_quantifier_dmrs)
                                            body_quantifier_dmrs.compose(body_dmrs, fusion={'body': 'rel'}, hierarchy=self.hierarchy)
                                            matches = list(dmrs_exact_matching(sub_dmrs=body_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                            if len(matches) >= 1:
                                                quantifier = Quantifier(qtype=qtype, qrange=qrange, quantity=quantity, restrictor=restrictor, body=body)
                                                number_bound = NumberBound(bound=bound, quantifier=quantifier)
                                                self.unused.discard(('number-bound', bound))
                                                yield number_bound, body_quantifier_dmrs

    def comparative_quantifier_caption(self, dmrs):
        for qtype in self.comparative_quantifiers:
            for qrange in self.comparative_quantifiers[qtype]:
                for quantity in self.comparative_quantifiers[qtype][qrange]:
                    quantifier_dmrs = self.comparative_quantifiers[qtype][qrange][quantity]
                    matches = list(dmrs_exact_matching(sub_dmrs=quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    for match in matches:
                        # print('comparative-quantifier > restrictor')
                        restrictor_nodeid = match[quantifier_dmrs.anchors['rstr'].nodeid]
                        restrictor_quantifier_nodeid = match[quantifier_dmrs.anchors['rquant'].nodeid]
                        comparison_nodeid = match[quantifier_dmrs.anchors['comp'].nodeid]
                        comparison_quantifier_nodeid = match[quantifier_dmrs.anchors['cquant'].nodeid]
                        body_nodeid = match[quantifier_dmrs.anchors['body'].nodeid]
                        unmatched_dmrs = dmrs.subgraph(nodeid=restrictor_nodeid, exclude=(comparison_nodeid, comparison_quantifier_nodeid, body_nodeid))  # dmrs.index.nodeid, dmrs.top.nodeid
                        for restrictor, restrictor_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                            rstr_quantifier_dmrs = copy.deepcopy(quantifier_dmrs)
                            rstr_quantifier_dmrs.compose(restrictor_dmrs, fusion={'rstr': 'type', 'rquant': 'quant'}, hierarchy=self.hierarchy)
                            matches = list(dmrs_exact_matching(sub_dmrs=rstr_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                            if len(matches) >= 1:
                                # print('comparative-quantifier > comparison')
                                for match in matches:
                                    unmatched_dmrs = dmrs.subgraph(nodeid=comparison_nodeid, exclude=(restrictor_nodeid, restrictor_quantifier_nodeid, body_nodeid))  # dmrs.index.nodeid, dmrs.top.nodeid
                                    unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in rstr_quantifier_dmrs if match[nodeid] not in (comparison_nodeid, comparison_quantifier_nodeid))
                                    for comparison, comparison_dmrs in self.type_caption(dmrs=unmatched_dmrs):
                                        comp_quantifier_dmrs = copy.deepcopy(rstr_quantifier_dmrs)
                                        comp_quantifier_dmrs.compose(comparison_dmrs, fusion={'comp': 'type', 'cquant': 'quant'}, hierarchy=self.hierarchy)
                                        matches = list(dmrs_exact_matching(sub_dmrs=comp_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                        if len(matches) >= 1:
                                            # print('comparative-quantifier > body')
                                            for match in matches:
                                                unmatched_dmrs = dmrs.subgraph(nodeid=body_nodeid, exclude=(restrictor_nodeid, restrictor_quantifier_nodeid, comparison_nodeid, comparison_quantifier_nodeid))  # dmrs.index.nodeid, dmrs.top.nodeid
                                                unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in comp_quantifier_dmrs if match[nodeid] != body_nodeid)
                                                for body, body_dmrs in self.relation_caption(dmrs=unmatched_dmrs):
                                                    body_quantifier_dmrs = copy.deepcopy(comp_quantifier_dmrs)
                                                    body_quantifier_dmrs.compose(body_dmrs, fusion={'body': 'rel'}, hierarchy=self.hierarchy)
                                                    matches = list(dmrs_exact_matching(sub_dmrs=body_quantifier_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                                                    if len(matches) >= 1:
                                                        comparative_quantifier = ComparativeQuantifier(qtype=qtype, qrange=qrange, quantity=quantity, restrictor=restrictor, comparison=comparison, body=body)
                                                        self.unused.discard(('comparative-quantifier', qtype, qrange, quantity))
                                                        yield comparative_quantifier, body_quantifier_dmrs

    def proposition_caption(self, dmrs):
        for proptype in self.propositions:
            if proptype in ('attribute', 'type', 'selector', 'relation', 'existential', 'quantifier', 'number-bound', 'comparative-quantifier'):
                continue
            proposition_dmrs = self.propositions[proptype]
            matches = list(dmrs_exact_matching(sub_dmrs=proposition_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('proposition > arg1')
                head_nodeid = match[proposition_dmrs.anchors['head'].nodeid]
                arg1_nodeid = match[proposition_dmrs.anchors['arg1'].nodeid]
                arg2_nodeid = match[proposition_dmrs.anchors['arg2'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=arg1_nodeid, exclude=(head_nodeid, arg2_nodeid))  # dmrs.index.nodeid, dmrs.top.nodeid
                for arg1, arg1_dmrs in self.caption_with_dmrs(dmrs=unmatched_dmrs):
                    arg1_proposition_dmrs = copy.deepcopy(proposition_dmrs)
                    arg1_proposition_dmrs.compose(arg1_dmrs, fusion={'arg1': 'head'}, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=arg1_proposition_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    for match in matches:
                        # print('proposition > arg2')
                        unmatched_dmrs = dmrs.subgraph(nodeid=arg2_nodeid, exclude=(head_nodeid, arg1_nodeid))  # dmrs.index.nodeid, dmrs.top.nodeid
                        unmatched_dmrs.remove_nodes(match[nodeid] for nodeid in arg1_proposition_dmrs if match[nodeid] != arg2_nodeid)
                        for arg2, arg2_dmrs in self.caption_with_dmrs(dmrs=unmatched_dmrs):
                            arg2_proposition_dmrs = copy.deepcopy(arg1_proposition_dmrs)
                            arg2_proposition_dmrs.compose(arg2_dmrs, fusion={'arg2': 'head'}, hierarchy=self.hierarchy)
                            matches = list(dmrs_exact_matching(sub_dmrs=arg2_proposition_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                            if len(matches) >= 1:
                                proposition = Proposition(proptype=proptype, clauses=(arg1, arg2))
                                self.unused.discard(('proposition', proptype))
                                yield proposition, arg2_proposition_dmrs

    def caption_with_dmrs(self, dmrs):
        yield from self.proposition_caption(dmrs=dmrs)

        if 'comparative-quantifier' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['comparative-quantifier'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > comparative-quantifier')
                quantifier_nodeid = match[self.propositions['comparative-quantifier'].anchors['head'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=quantifier_nodeid)  # , dmrs.top.nodeid
                for comparative_quantifier, quantifier_dmrs in self.comparative_quantifier_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['comparative-quantifier'])
                    caption_dmrs.compose(quantifier_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'comparative-quantifier'))
                        yield comparative_quantifier, caption_dmrs

        if 'number-bound' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['number-bound'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > number-bound')
                bound_nodeid = match[self.propositions['number-bound'].anchors['head'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=bound_nodeid)  # , dmrs.top.nodeid
                for number_bound, bound_dmrs in self.number_bound_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['number-bound'])
                    caption_dmrs.compose(bound_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'number-bound'))
                        yield number_bound, caption_dmrs

        if 'quantifier' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['quantifier'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > quantifier')
                quantifier_nodeid = match[self.propositions['quantifier'].anchors['head'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=quantifier_nodeid)  # , dmrs.top.nodeid
                for quantifier, quantifier_dmrs in self.quantifier_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['quantifier'])
                    caption_dmrs.compose(quantifier_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'quantifier'))
                        yield quantifier, caption_dmrs

        if 'existential' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['existential'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > existential')
                existential_nodeid = match[self.propositions['existential'].anchors['head'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=existential_nodeid)  # , dmrs.top.nodeid
                for existential, existential_dmrs in self.existential_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['existential'])
                    caption_dmrs.compose(existential_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'existential'))
                        yield existential, caption_dmrs

        if 'relation' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['relation'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > relation')
                relation_nodeid = match[self.propositions['relation'].anchors['rel'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=relation_nodeid)  # , dmrs.top.nodeid
                for relation, relation_dmrs in self.relation_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['relation'])
                    caption_dmrs.compose(relation_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'relation'))
                        yield relation, caption_dmrs

        if 'selector' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['selector'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > selector')
                scope_nodeid = match[self.propositions['selector'].anchors['scope'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=scope_nodeid)  # , dmrs.top.nodeid
                for selector, selector_dmrs in self.selector_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['selector'])
                    caption_dmrs.compose(selector_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'selector'))
                        yield selector, caption_dmrs

        if 'type' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['type'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > type')
                type_nodeid = match[self.propositions['type'].anchors['type'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=type_nodeid)  # , dmrs.top.nodeid
                for entity_type, type_dmrs in self.type_caption(dmrs=unmatched_dmrs):  # only one?
                    caption_dmrs = copy.deepcopy(self.propositions['type'])
                    caption_dmrs.compose(type_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'type'))
                        yield entity_type, caption_dmrs

        if 'attribute' in self.propositions:
            matches = list(dmrs_exact_matching(sub_dmrs=self.propositions['attribute'], dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
            for match in matches:
                # print('caption > attribute')
                attribute_nodeid = match[self.propositions['attribute'].anchors['type'].nodeid]
                unmatched_dmrs = dmrs.subgraph(nodeid=attribute_nodeid)  # , dmrs.top.nodeid
                for attribute, attribute_dmrs in self.attribute_caption(dmrs=unmatched_dmrs):
                    caption_dmrs = copy.deepcopy(self.propositions['attribute'])
                    caption_dmrs.compose(attribute_dmrs, hierarchy=self.hierarchy)
                    matches = list(dmrs_exact_matching(sub_dmrs=caption_dmrs, dmrs=dmrs, hierarchy=self.hierarchy, match_top_index=False))
                    if len(matches) == 1:
                        self.unused.discard(('proposition', 'attribute'))
                        yield attribute, caption_dmrs
