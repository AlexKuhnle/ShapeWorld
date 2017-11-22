import copy
import json
import os
import re
import subprocess
from shapeworld import util
from shapeworld.captions import Attribute, Relation, EntityType, Existential, Quantifier, NumberBound, ComparativeQuantifier, Proposition
from shapeworld.realizers import CaptionRealizer
from shapeworld.realizers.dmrs.dmrs import Dmrs, create_sortinfo


def prepare_grammar(language):
    # dmrs sub-directory
    directory = os.path.dirname(os.path.realpath(__file__))
    # unpack grammar if used for the first time
    if not os.path.isfile(os.path.join(directory, 'languages', language + '.dat')):
        assert os.path.isfile(os.path.join(directory, 'languages', language + '.dat.gz'))
        import gzip
        with gzip.open(os.path.join(directory, 'languages', language + '.dat.gz'), 'rb') as gzip_filehandle:
            with open(os.path.join(directory, 'languages', language + '.dat'), 'wb') as filehandle:
                filehandle.write(gzip_filehandle.read())


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


class DmrsRealizer(CaptionRealizer):

    def __init__(self, language):
        super(DmrsRealizer, self).__init__(language)
        prepare_grammar(language=language)
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.ace_path = os.path.join(directory, 'resources', 'ace')
        self.erg_path = os.path.join(directory, 'languages', language + '.dat')

        # self.regex = re.compile(pattern=r'^(NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\])|(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)|(WARNING: unknown lexeme \'be_inv_am\'!)|(ERROR: trigger rules call for non-existant lexeme `be_inv_am\')$')
        self.successful_regex = re.compile(pattern=r'^NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\]$')
        self.unsuccessful_regex = re.compile(pattern=r'^NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[0 results\]$')
        self.final_regex = re.compile(pattern=r'^(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)$')

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

        self.attributes = dict()
        self.attribute_by_key = dict()
        self.empty_type = None
        self.relation_attribute = None
        if 'attributes' in language:
            for predtype, values in language['attributes'].items():
                predtype = parse_string(predtype)
                if predtype == 'empty':
                    self.empty_type = Dmrs.parse(values['dmrs'])
                    continue
                elif predtype == 'relation':
                    self.relation_attribute = Dmrs.parse(values['dmrs'])
                    continue
                elif predtype not in self.attributes:
                    self.attributes[predtype] = dict()
                for value, attribute in values.items():
                    value = parse_string(value)
                    self.attributes[predtype][value] = Dmrs.parse(attribute['dmrs'])
                    assert str(attribute['key']) not in self.attribute_by_key
                    self.attribute_by_key[str(attribute['key'])] = (predtype, value)

        self.relations = dict()
        self.relation_by_key = dict()
        self.attribute_relation = None
        self.type_relation = None
        if 'relations' in language:
            for predtype, values in language['relations'].items():
                predtype = parse_string(predtype)
                if predtype == 'attribute':
                    self.attribute_relation = Dmrs.parse(values['dmrs'])
                    continue
                elif predtype == 'type':
                    self.type_relation = Dmrs.parse(values['dmrs'])
                    continue
                elif predtype not in self.relations:
                    self.relations[predtype] = dict()
                for value, relation in values.items():
                    value = parse_string(value)
                    self.relations[predtype][value] = Dmrs.parse(relation['dmrs'])
                    assert str(relation['key']) not in self.relation_by_key
                    self.relation_by_key[str(relation['key'])] = (predtype, value)

        self.existential = None
        if 'existential' in language:
            self.existential = Dmrs.parse(language['existential']['dmrs'])

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
                    continue
                for qrange, quantities in qranges.items():
                    qrange = parse_string(qrange)
                    if qrange not in self.quantifiers[qtype]:
                        self.quantifiers[qtype][qrange] = dict()
                    for quantity, quantifier in quantities.items():
                        quantity = parse_string(quantity)
                        self.quantifiers[qtype][qrange][quantity] = Dmrs.parse(quantifier['dmrs'])
                        assert str(quantifier['key']) not in self.quantifier_by_key
                        self.quantifier_by_key[str(quantifier['key'])] = (qtype, qrange, quantity)

        self.number_bounds = dict()
        self.number_bound_by_key = dict()
        if 'number-bounds' in language:
            for bound, number_bound in language['number-bounds'].items():
                bound = parse_string(bound)
                self.number_bounds[bound] = Dmrs.parse(number_bound['dmrs'])
                assert str(number_bound['key']) not in self.number_bound_by_key
                self.number_bound_by_key[str(number_bound['key'])] = (bound,)

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
                    continue
                for qrange, quantities in qranges.items():
                    qrange = parse_string(qrange)
                    if qrange not in self.comparative_quantifiers[qtype]:
                        self.comparative_quantifiers[qtype][qrange] = dict()
                    for quantity, quantifier in quantities.items():
                        quantity = parse_string(quantity)
                        self.comparative_quantifiers[qtype][qrange][quantity] = Dmrs.parse(quantifier['dmrs'])
                        assert str(quantifier['key']) not in self.comparative_quantifier_by_key
                        self.comparative_quantifier_by_key[str(quantifier['key'])] = (qtype, qrange, quantity)

        self.propositions = dict()
        self.proposition_by_key = dict()
        for connective, proposition in language['propositions'].items():
            connective = parse_string(connective)
            if isinstance(proposition['dmrs'], list):
                self.propositions[connective] = tuple(Dmrs.parse(dmrs) for dmrs in proposition['dmrs'])
            else:
                self.propositions[connective] = Dmrs.parse(proposition['dmrs'])
            assert str(proposition['key']) not in self.proposition_by_key
            self.proposition_by_key[str(proposition['key'])] = connective

        self.hierarchy = language['hierarchy']

        self.post_processing = dict()
        for key, paraphrase in language['post-processing'].items():
            search = Dmrs.parse(paraphrase['search'])
            replace = Dmrs.parse(paraphrase['replace'])
            self.post_processing[str(key)] = (search, replace)

    def realize(self, captions):
        try:
            ace = subprocess.Popen([self.ace_path, '-g', self.erg_path, '-1e', '-r', 'root_strict'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            import sys
            from datetime import datetime
            print(datetime.now().strftime('%H:%M:%S'))
            print(e.strerror)
            print(sys.exc_info()[0])
            raise
        dmrs_list = list()
        mrs_list = list()
        for caption in captions:
            dmrs = self.caption_dmrs(caption=caption)
            dmrs = dmrs.apply_paraphrases(self.post_processing.values())
            dmrs.remove_underspecifications()
            dmrs_list.append(dmrs)
            mrs_list.append(dmrs.get_mrs() + '\n')
        stdout_data, stderr_data = ace.communicate(input=''.join(mrs_list).encode())
        stderr_data = stderr_data.decode('utf-8').splitlines()
        stdout_data = stdout_data.decode('utf-8').splitlines()

        failures = 0
        n = 0
        unexpected = False
        for line in stderr_data:
            if n == len(captions):
                assert self.final_regex.match(line), line
                continue
            if self.successful_regex.match(line):
                if unexpected:
                    print(dmrs_list[n].dumps_xml().decode())
                    print(mrs_list[n])
                    unexpected = False
                n += 1
            elif self.unsuccessful_regex.match(line):
                print(dmrs_list[n].dumps_xml().decode())
                print(mrs_list[n])
                failures += 1
                n += 1
            else:
                print('Unexpected: ' + line)
                unexpected = True
        if failures > 0:
            print('Failures: {}'.format(failures))
            exit(0)

        caption_strings = [line for line in stdout_data if line]
        assert len(caption_strings) == len(captions), stdout_data + '\n' + stderr_data
        for n, caption in enumerate(caption_strings):
            captions[n] = util.string2tokens(string=caption)
        return captions

    def attribute_dmrs(self, attribute):
        if attribute.predtype == 'relation':
            assert self.relation_attribute is not None
            dmrs = copy.deepcopy(self.relation_attribute)
            dmrs.compose(self.relation_dmrs(attribute.value), fusion={'attr': 'rel'}, hierarchy=self.hierarchy)
        else:
            assert attribute.predtype in self.attributes and attribute.value in self.attributes[attribute.predtype], (attribute.predtype, attribute.value)
            dmrs = copy.deepcopy(self.attributes[attribute.predtype][attribute.value])
        return dmrs

    def type_dmrs(self, etype):
        assert self.empty_type is not None
        dmrs = copy.deepcopy(self.empty_type)
        for attribute in etype.value.values():
            dmrs.compose(self.attribute_dmrs(attribute), fusion={'type': 'type'}, hierarchy=self.hierarchy)
        return dmrs

    def relation_dmrs(self, relation):
        if relation.predtype == 'attribute':
            assert self.attribute_relation is not None
            dmrs = copy.deepcopy(self.attribute_relation)
            dmrs.compose(self.attribute_dmrs(relation.value), fusion={'ref': 'type'}, hierarchy=self.hierarchy)
        elif relation.predtype == 'type':
            assert self.type_relation is not None
            dmrs = copy.deepcopy(self.type_relation)
            dmrs.compose(self.type_dmrs(relation.value), fusion={'ref': 'type'}, hierarchy=self.hierarchy)
        else:
            assert relation.predtype in self.relations and relation.value in self.relations[relation.predtype], (relation.predtype, relation.value)
            dmrs = copy.deepcopy(self.relations[relation.predtype][relation.value])
            dmrs.compose(self.type_dmrs(relation.reference), fusion={'ref': 'type'}, hierarchy=self.hierarchy)
            if relation.predtype in Relation.ternary_relations:
                dmrs.compose(self.type_dmrs(relation.comparison), fusion={'comp': 'type'}, hierarchy=self.hierarchy)
        return dmrs

    def existential_dmrs(self, existential):
        assert self.existential is not None
        dmrs = copy.deepcopy(self.existential)
        dmrs.compose(self.type_dmrs(existential.restrictor), fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(existential.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def quantifier_dmrs(self, quantifier):
        assert quantifier.qtype in self.quantifiers and quantifier.qrange in self.quantifiers[quantifier.qtype] and quantifier.quantity in self.quantifiers[quantifier.qtype][quantifier.qrange], (quantifier.qtype, quantifier.qrange, quantifier.quantity)
        dmrs = copy.deepcopy(self.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity])
        dmrs.compose(self.type_dmrs(quantifier.restrictor), fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(quantifier.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def number_bound_dmrs(self, number_bound):
        assert number_bound.bound in self.number_bounds, number_bound.bound
        # TODO: Would be more compositional
        # dmrs = copy.deepcopy(self.number_bounds[number_bound.bound])
        # dmrs.compose(self.quantifier_dmrs(number_bound.quantifier), fusion={'rstr': 'rstr'}, other_head=True, hierarchy=self.hierarchy)
        # Better: being able to swap nodes
        # "2": {"key": "of_the_two", "dmrs": "_the_q --> [type]:part_of x[pers=3] <=1= card(2) e; [rstr]:pred x? -1-> :part_of"},
        quantifier = number_bound.quantifier
        assert quantifier.qtype in self.quantifiers and quantifier.qrange in self.quantifiers[quantifier.qtype] and quantifier.quantity in self.quantifiers[quantifier.qtype][quantifier.qrange]
        rstr_dmrs = copy.deepcopy(self.number_bounds[number_bound.bound])
        rstr_dmrs.compose(self.type_dmrs(quantifier.restrictor), fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs = copy.deepcopy(self.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity])
        dmrs.compose(rstr_dmrs, fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(quantifier.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def comparative_quantifier_dmrs(self, comparative_quantifier):
        assert comparative_quantifier.qtype in self.comparative_quantifiers and comparative_quantifier.qrange in self.comparative_quantifiers[comparative_quantifier.qtype] and comparative_quantifier.quantity in self.comparative_quantifiers[comparative_quantifier.qtype][comparative_quantifier.qrange], (comparative_quantifier.qtype, comparative_quantifier.qrange, comparative_quantifier.quantity)
        dmrs = copy.deepcopy(self.comparative_quantifiers[comparative_quantifier.qtype][comparative_quantifier.qrange][comparative_quantifier.quantity])
        dmrs.compose(self.type_dmrs(comparative_quantifier.restrictor), fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.type_dmrs(comparative_quantifier.comparison), fusion={'comp': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(comparative_quantifier.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def proposition_dmrs(self, proposition):
        assert proposition.proptype in self.propositions, proposition.proptype
        dmrs = copy.deepcopy(self.propositions[proposition.proptype])
        clauses_dmrs = []
        for clause in proposition.clauses:
            clause_dmrs = self.caption_dmrs(clause)
            clauses_dmrs.append(clause_dmrs)
        if isinstance(dmrs, Dmrs):
            intermediate_dmrs = dmrs
            first_dmrs = None
        else:
            dmrs, intermediate_dmrs, first_dmrs = dmrs
        dmrs.compose(clauses_dmrs[-1], fusion={'arg2': 'head'}, hierarchy=self.hierarchy)
        for clause_dmrs in reversed(clauses_dmrs[1:-1]):
            dmrs.compose(clause_dmrs, fusion={'arg1': 'head'}, hierarchy=self.hierarchy)
            dmrs.compose(intermediate_dmrs, fusion={'arg2': 'con'}, other_head=True, hierarchy=self.hierarchy)
        dmrs.compose(clauses_dmrs[0], fusion={'arg1': 'head'}, hierarchy=self.hierarchy)
        if first_dmrs:
            dmrs.compose(first_dmrs, fusion={'arg1': 'arg'}, hierarchy=self.hierarchy)
        return dmrs

    def caption_dmrs(self, caption):
        if isinstance(caption, Attribute):
            dmrs = copy.deepcopy(self.propositions['attribute'])
            dmrs.compose(self.attribute_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, EntityType):
            dmrs = copy.deepcopy(self.propositions['type'])
            dmrs.compose(self.type_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, Relation):
            dmrs = copy.deepcopy(self.propositions['relation'])
            dmrs.compose(self.relation_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, Existential):
            dmrs = copy.deepcopy(self.propositions['existential'])
            dmrs.compose(self.existential_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, Quantifier):
            dmrs = copy.deepcopy(self.propositions['quantifier'])
            dmrs.compose(self.quantifier_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, NumberBound):
            dmrs = copy.deepcopy(self.propositions['number_bound'])
            dmrs.compose(self.number_bound_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, ComparativeQuantifier):
            dmrs = copy.deepcopy(self.propositions['comparative_quantifier'])
            dmrs.compose(self.comparative_quantifier_dmrs(caption), hierarchy=self.hierarchy)
        elif isinstance(caption, Proposition):
            dmrs = self.proposition_dmrs(caption)
        else:
            assert False
        return dmrs


realizer = DmrsRealizer


"""
        "proximity-rel": {
            "-1": {"key": "closer", "dmrs": "[rel]:_close_a_to e? <=1= _to_p e -2-> [ref]:node <-- _a_q; :rel <=1= comp e -2-> [comp]:node <-- _a_q"},
            "1": {"key": "farther", "dmrs": "[rel]:_far_a_from e? <=1= _from_p e -2-> [ref]:node <-- _a_q; :rel <=1= comp e -2-> [comp]:node <-- _a_q"}
        },
"""
