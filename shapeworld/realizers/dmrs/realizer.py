import copy
import json
import os
import re
import subprocess
from shapeworld import util
from shapeworld.caption import Attribute, Relation, EntityType, Existential, Quantifier, Proposition
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
tuple_regex = re.compile(pattern=r'^[a-z]+(,[a-z]+)+$')


def parse_string(string):
    if int_regex.match(string):
        return int(string)
    elif float_regex.match(string):
        return float(string)
    elif tuple_regex.match(string):
        return tuple(string)
    else:
        return string


class DmrsRealizer(CaptionRealizer):

    def __init__(self, language):
        super(DmrsRealizer, self).__init__(language)
        prepare_grammar(language=language)
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.ace_path = os.path.join(directory, 'resources', 'ace')
        self.erg_path = os.path.join(directory, 'languages', language + '.dat')
        self.regex = re.compile(pattern=r'^(NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\])|(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)$')

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

        self.attributes = {attrtype: dict() for attrtype in language['attributes']}
        self.attribute_by_name = dict()
        for attrtype, values in language['attributes'].items():
            if attrtype == 'empty' or attrtype == 'relation':
                values['dmrs'] = Dmrs.parse(values['dmrs'])
                self.attributes[attrtype] = values
                continue
            for value, attribute in values.items():
                value = parse_string(value)
                attribute['dmrs'] = Dmrs.parse(attribute['dmrs'])
                self.attributes[attrtype][value] = attribute
                self.attribute_by_name[attribute['key']] = (attrtype, value)

        self.relations = {reltype: dict() for reltype in language['relations']}
        self.relation_by_name = dict()
        for reltype, values in language['relations'].items():
            if reltype == 'attribute' or reltype == 'type':
                values['dmrs'] = Dmrs.parse(values['dmrs'])
                self.relations[reltype] = values
                continue
            for value, relation in values.items():
                value = parse_string(value)
                relation['dmrs'] = Dmrs.parse(relation['dmrs'])
                self.relations[reltype][value] = relation
                self.relation_by_name[relation['key']] = (reltype, value)

        self.quantifiers = {qtype: {qrange: dict() for qrange in qranges} for qtype, qranges in language['quantifiers'].items()}
        self.quantifier_by_name = dict()
        for qtype, qranges in language['quantifiers'].items():
            if qtype == 'existential':
                qranges['dmrs'] = Dmrs.parse(qranges['dmrs'])
                self.quantifiers[qtype] = qranges
                continue
            for qrange, quantities in qranges.items():
                qrange = parse_string(qrange)
                for quantity, quantifier in quantities.items():
                    quantity = parse_string(quantity)
                    quantifier['dmrs'] = Dmrs.parse(quantifier['dmrs'])
                    self.quantifiers[qtype][qrange][quantity] = quantifier
                    self.quantifier_by_name[quantifier['key']] = (qtype, qrange, quantity)

        self.propositions = dict()
        self.proposition_by_name = dict()
        for connective, proposition in language['propositions'].items():
            if connective in ('attribute', 'type', 'relation', 'existential', 'quantifier'):
                proposition['dmrs'] = Dmrs.parse(proposition['dmrs'])
                self.propositions[connective] = proposition
                continue
            if isinstance(proposition['dmrs'], list):
                proposition['dmrs'] = tuple(Dmrs.parse(dmrs) for dmrs in proposition['dmrs'])
            else:
                proposition['dmrs'] = Dmrs.parse(proposition['dmrs'])
            self.propositions[connective] = proposition
            self.proposition_by_name[proposition['key']] = connective

        self.hierarchy = language['hierarchy']

        self.post_processing = dict()
        for key, paraphrase in language['post-processing'].items():
            search = Dmrs.parse(paraphrase['search'])
            replace = Dmrs.parse(paraphrase['replace'])
            self.post_processing[key] = (search, replace)

    def realize(self, captions):
        try:
            ace = subprocess.Popen([self.ace_path, '-g', self.erg_path, '-1Te', '-r', 'root_strict'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            dmrs = self.clause_dmrs(caption)
            dmrs.apply_paraphrases(self.post_processing.values())
            dmrs.remove_underspecifications()
            dmrs_list.append(dmrs)
            mrs_list.append(dmrs.get_mrs() + '\n')
        stdout_data, stderr_data = ace.communicate(''.join(mrs_list).encode())
        stderr_data = stderr_data.decode('utf-8').splitlines()
        stdout_data = stdout_data.decode('utf-8').splitlines()
        assert all(self.regex.match(line) for line in stderr_data), '\n\n' + '\n'.join('{}\n{}\n{}\n'.format(line, dmrs.dumps_xml().decode(), mrs) for line, dmrs, mrs in zip(stderr_data, dmrs_list, mrs_list) if not self.regex.match(line)) + '\nFailures: {}\n'.format(len(captions) - int(stderr_data[-2][16:stderr_data[-2].index(' ', 16)]))  # self.proposition_dmrs(caption).dumps_xml()
        caption_strings = [line for line in stdout_data if line]
        assert len(caption_strings) == len(captions)
        for n, caption in enumerate(caption_strings):
            captions[n] = util.string2tokens(string=caption)
        return captions

    def component_dmrs(self, component, key=False):
        assert 'dmrs' in component
        if key:
            assert 'key' in component
            return component['key'], copy.deepcopy(component['dmrs'])  # choice if list?
        else:
            return copy.deepcopy(component['dmrs'])  # choice if list?

    def attribute_dmrs(self, attribute):
        assert attribute.attrtype in self.attributes
        if attribute.attrtype == 'relation':
            dmrs = self.component_dmrs(self.attributes['relation'])
            dmrs.compose(self.relation_dmrs(attribute.value), fusion={'attr': 'rel'}, hierarchy=self.hierarchy)
        else:
            assert attribute.value in self.attributes[attribute.attrtype]
            dmrs = self.component_dmrs(self.attributes[attribute.attrtype][attribute.value])
        return dmrs

    def type_dmrs(self, etype):
        assert 'empty' in self.attributes
        dmrs = self.component_dmrs(self.attributes['empty'])
        for attribute in etype.attributes:
            dmrs.compose(self.attribute_dmrs(attribute), fusion={'type': 'type'}, hierarchy=self.hierarchy)
        return dmrs

    def relation_dmrs(self, relation):
        assert relation.reltype in self.relations
        if relation.reltype == 'attribute':
            dmrs = self.component_dmrs(self.relations['attribute'])
            dmrs.compose(self.attribute_dmrs(relation.value), fusion={'ref': 'type'}, hierarchy=self.hierarchy)
        elif relation.reltype == 'type':
            dmrs = self.component_dmrs(self.relations['type'])
            dmrs.compose(self.type_dmrs(relation.value), fusion={'ref': 'type'}, hierarchy=self.hierarchy)
        else:
            assert relation.value in self.relations[relation.reltype]
            dmrs = self.component_dmrs(self.relations[relation.reltype][relation.value])
            dmrs.compose(self.type_dmrs(relation.reference), fusion={'ref': 'type'}, hierarchy=self.hierarchy)
            if relation.reltype in Relation.ternary_relations:
                dmrs.compose(self.type_dmrs(relation.comparison), fusion={'comp': 'type'}, hierarchy=self.hierarchy)
        return dmrs

    def existential_dmrs(self, existential):
        assert 'existential' in self.quantifiers
        dmrs = self.component_dmrs(self.quantifiers['existential'])
        dmrs.compose(self.type_dmrs(existential.restrictor), fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(existential.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def quantifier_dmrs(self, quantifier):
        assert quantifier.qtype in self.quantifiers and quantifier.qrange in self.quantifiers[quantifier.qtype] and quantifier.quantity in self.quantifiers[quantifier.qtype][quantifier.qrange]
        dmrs = self.component_dmrs(self.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity])
        dmrs.compose(self.type_dmrs(quantifier.restrictor), fusion={'rstr': 'type'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(quantifier.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def proposition_dmrs(self, proposition):
        assert proposition.proptype in self.propositions
        dmrs = self.component_dmrs(self.propositions[proposition.proptype])
        clauses_dmrs = []
        for clause in proposition.clauses:
            clause_dmrs = self.clause_dmrs(clause)
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

    def clause_dmrs(self, clause):
        if isinstance(clause, Attribute):
            dmrs = self.component_dmrs(self.propositions['attribute'])
            dmrs.compose(self.attribute_dmrs(clause), hierarchy=self.hierarchy)
        elif isinstance(clause, EntityType):
            dmrs = self.component_dmrs(self.propositions['type'])
            dmrs.compose(self.type_dmrs(clause), hierarchy=self.hierarchy)
        elif isinstance(clause, Relation):
            dmrs = self.component_dmrs(self.propositions['relation'])
            dmrs.compose(self.relation_dmrs(clause), hierarchy=self.hierarchy)
        elif isinstance(clause, Existential):
            dmrs = self.component_dmrs(self.propositions['existential'])
            dmrs.compose(self.existential_dmrs(clause), hierarchy=self.hierarchy)
        elif isinstance(clause, Quantifier):
            dmrs = self.component_dmrs(self.propositions['quantifier'])
            dmrs.compose(self.quantifier_dmrs(clause), hierarchy=self.hierarchy)
        elif isinstance(clause, Proposition):
            dmrs = self.proposition_dmrs(clause)
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
