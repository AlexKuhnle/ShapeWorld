from copy import deepcopy
import json
import os
import re
import subprocess
from shapeworld import CaptionRealizer
from shapeworld.caption import Modifier, Relation, Noun, Existential, Quantifier, Proposition
from shapeworld.realizers.dmrs.dmrs import Dmrs, event_sortinfo, instance_sortinfo


def prepare_grammar(language):
    # dmrs sub-directory
    directory = os.path.dirname(os.path.realpath(__file__))
    # unpack grammar if used for the first time
    if not os.path.isfile(os.path.join(directory, 'resources', language + '.dat')):
        assert os.path.isfile(os.path.join(directory, 'resources', language + '.dat.tar.gz'))
        import tarfile
        with tarfile.open(os.path.join(directory, 'resources', language + '.dat.tar.gz'), 'r:gz') as filehandle:
            try:
                fileinfo = filehandle.getmember(language + '.dat')
            except KeyError:
                assert False
            filehandle.extract(member=fileinfo)


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
        prepare_grammar(language=language)
        self.language = language
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
        self.ace_path = os.path.join(directory, 'resources', 'ace')
        self.erg_path = os.path.join(directory, 'resources', language + '.dat')
        self.regex = re.compile(pattern=r'^(NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[1 results\])|(NOTE: generated [0-9]+ / [0-9]+ sentences, avg [0-9]+k, time [0-9]+.[0-9]+s)|(NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones)$')

        with open(os.path.join(directory, 'languages', language + '.json'), 'r') as filehandle:
            language = json.load(fp=filehandle)

        if 'event_features' in language:
            self.event_sortinfo = event_sortinfo(features=language['event_features'])
        else:
            self.event_sortinfo = None
        if 'instance_features' in language:
            self.instance_sortinfo = instance_sortinfo(features=language['instance_features'])
        else:
            self.instance_sortinfo = None

        self.modifiers = {modtype: dict() for modtype in language['modifiers']}
        self.modifier_by_name = dict()
        for modtype, values in language['modifiers'].items():
            if modtype == 'empty' or modtype == 'relation':
                values['dmrs'] = Dmrs.parse(values['dmrs'])
                self.modifiers[modtype] = values
                continue
            for value, modifier in values.items():
                value = parse_string(value)
                modifier['dmrs'] = Dmrs.parse(modifier['dmrs'])
                self.modifiers[modtype][value] = modifier
                self.modifier_by_name[modifier['key']] = (modtype, value)

        # self.nouns = {modtype: dict() for modtype in language['nouns']}
        # self.noun_by_name = dict()
        # for modtype, values in language['nouns'].items():
        #     if modtype == 'empty':
        #         values['dmrs'] = Dmrs.parse(values['dmrs'])
        #         self.nouns[modtype] = values
        #         continue
        #     for value, noun in values.items():
        #         value = parse_string(value)
        #         noun['dmrs'] = Dmrs.parse(noun['dmrs'])
        #         self.nouns[modtype][value] = noun
        #         self.noun_by_name[noun['key']] = (modtype, value)

        self.relations = {reltype: dict() for reltype in language['relations']}
        self.relation_by_name = dict()
        for reltype, values in language['relations'].items():
            if reltype == 'modifier' or reltype == 'noun':
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

        # self.singular_quantifiers = set(language['singular-quantifiers'])

        self.propositions = dict()
        self.proposition_by_name = dict()
        for connective, proposition in language['propositions'].items():
            if connective in ('modifier', 'noun', 'relation', 'existential', 'quantifier'):
                proposition['dmrs'] = Dmrs.parse(proposition['dmrs'])
                self.propositions[connective] = proposition
                continue
            if isinstance(proposition['dmrs'], str):
                proposition['dmrs'] = Dmrs.parse(proposition['dmrs'])
            else:
                proposition['dmrs'] = tuple(Dmrs.parse(dmrs) for dmrs in proposition['dmrs'])
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
            caption = caption.lower()
            caption = caption.replace('.', ' .').replace(',', ' ,')
            captions[n] = caption.split()
        return captions

    def component_dmrs(self, component, key=False):
        assert 'dmrs' in component
        if key:
            assert 'key' in component
            return component['key'], deepcopy(component['dmrs'])  # choice if list?
        else:
            return deepcopy(component['dmrs'])  # choice if list?

    def modifier_dmrs(self, modifier):
        assert modifier.modtype in self.modifiers
        if modifier.modtype == 'relation':
            dmrs = self.component_dmrs(self.modifiers['relation'])
            dmrs.compose(self.relation_dmrs(modifier.value), fusion={'mod': 'rel'}, hierarchy=self.hierarchy)
        else:
            assert modifier.value in self.modifiers[modifier.modtype]
            dmrs = self.component_dmrs(self.modifiers[modifier.modtype][modifier.value])
        return dmrs

    def noun_dmrs(self, noun):
        assert 'empty' in self.modifiers
        dmrs = self.component_dmrs(self.modifiers['empty'])
        for modifier in noun.modifiers:
            dmrs.compose(self.modifier_dmrs(modifier), fusion={'entity': 'arg'}, hierarchy=self.hierarchy)
        return dmrs

    def relation_dmrs(self, relation):
        assert relation.reltype in self.relations
        if relation.reltype == 'modifier':
            dmrs = self.component_dmrs(self.relations['modifier'])
            dmrs.compose(self.modifier_dmrs(relation.value), fusion={'ref': 'mod'}, hierarchy=self.hierarchy)
        elif relation.reltype == 'noun':
            dmrs = self.component_dmrs(self.relations['noun'])
            dmrs.compose(self.noun_dmrs(relation.value), fusion={'ref': 'entity'}, hierarchy=self.hierarchy)
        else:
            assert relation.value in self.relations[relation.reltype]
            dmrs = self.component_dmrs(self.relations[relation.reltype][relation.value])
            dmrs.compose(self.noun_dmrs(relation.reference), fusion={'ref': 'entity'}, hierarchy=self.hierarchy)
            if relation.comparison is not None:
                dmrs.compose(self.noun_dmrs(relation.comparison), fusion={'comp': 'entity'}, hierarchy=self.hierarchy)
        return dmrs

    def existential_dmrs(self, existential):
        assert 'existential' in self.quantifiers
        dmrs = self.component_dmrs(self.quantifiers['existential'])
        dmrs.compose(self.noun_dmrs(existential.subject), fusion={'subj': 'entity'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(existential.verb), fusion={'verb': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def quantifier_dmrs(self, quantifier):
        assert quantifier.qtype in self.quantifiers and quantifier.qrange in self.quantifiers[quantifier.qtype] and quantifier.quantity in self.quantifiers[quantifier.qtype][quantifier.qrange]
        dmrs = self.component_dmrs(self.quantifiers[quantifier.qtype][quantifier.qrange][quantifier.quantity])
        dmrs.compose(self.noun_dmrs(quantifier.restrictor), fusion={'rstr': 'entity'}, hierarchy=self.hierarchy)
        dmrs.compose(self.relation_dmrs(quantifier.body), fusion={'body': 'rel'}, hierarchy=self.hierarchy)
        return dmrs

    def proposition_dmrs(self, proposition):
        assert proposition.proptype in self.propositions
        dmrs = self.component_dmrs(self.propositions[proposition.proptype])
        if proposition.proptype == 'modifier':
            dmrs.compose(self.modifier_dmrs(proposition.clauses[0]), hierarchy=self.hierarchy)
        elif proposition.proptype == 'noun':
            dmrs.compose(self.noun_dmrs(proposition.clauses[0]), hierarchy=self.hierarchy)
        elif proposition.proptype == 'relation':
            dmrs.compose(self.relation_dmrs(proposition.clauses[0]), hierarchy=self.hierarchy)
        elif proposition.proptype == 'existential':
            dmrs.compose(self.existential_dmrs(proposition.clauses[0]), hierarchy=self.hierarchy)
        elif proposition.proptype == 'quantifier':
            dmrs.compose(self.quantifier_dmrs(proposition.clauses[0]), hierarchy=self.hierarchy)
        else:
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
        if isinstance(clause, Modifier):
            dmrs = self.component_dmrs(self.propositions['modifier'])
            dmrs.compose(self.modifier_dmrs(clause), hierarchy=self.hierarchy)
        elif isinstance(clause, Noun):
            dmrs = self.component_dmrs(self.propositions['noun'])
            dmrs.compose(self.noun_dmrs(clause), hierarchy=self.hierarchy)
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
