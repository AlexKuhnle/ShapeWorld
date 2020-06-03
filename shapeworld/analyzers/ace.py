import argparse
import re
import subprocess
import sys
from shapeworld.realizers.dmrs.pydmrs.pydmrs.core import ListDmrs
from shapeworld.analyzers.mrs import Mrs
from shapeworld.analyzers.derivtree import DerivTree


class Ace:

    # root_gen, root_informal, root_strict
    def __init__(self, executable='ace', grammar='erg.dat', num_outputs=None, trees=False, root=None, informal=False):
        self.args = [executable, '-g', grammar]
        if num_outputs is not None:
            assert num_outputs > 0
            if num_outputs == 1:
                self.args.append('-1')
            else:
                self.args.append('-n')
                self.args.append(str(num_outputs))
        self.first_only = (num_outputs == 1)
        if not trees:
            self.args.append('-T')
        self.trees = trees
        if root is not None:
            self.args.append('-r')
            self.args.append(root)
        elif informal:
            self.args.append('-r')
            self.args.append('root_informal')

        # parsing regexes
        self.parse_successful_regex = re.compile(pattern=r'^NOTE: [1-9][0-9]* readings, added [0-9]+ / [0-9]+ edges to chart \([0-9]+ fully instantiated, [0-9]+ actives used, [0-9]+ passives used\)\tRAM: [0-9]+k$')
        self.parse_unsuccessful_regex = re.compile(pattern=r'(^NOTE: 0 readings, added [0-9]+ / [0-9]+ edges to chart \([0-9]+ fully instantiated, [0-9]+ actives used, [0-9]+ passives used\)\tRAM: [0-9]+k$)|(^NOTE: ignore$)')
        self.parse_final0_regex = re.compile(pattern=r'^NOTE: parsed [0-9]+ / 0 sentences, avg [0-9]+k, time [0-9]+.[0-9]+s$')
        self.parse_final1_regex = re.compile(pattern=r'^NOTE: parsed [0-9]+ / [1-9][0-9]* sentences, avg [0-9]+k, time [0-9]+.[0-9]+s$')

        # generation regexes
        self.gen_successful_regex = re.compile(pattern=r'^NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[[1-9][0-9]* results\]$')
        self.gen_unsuccessful_regex = re.compile(pattern=r'^NOTE: [0-9]+ passive, [0-9]+ active edges in final generation chart; built [0-9]+ passives total. \[0 results\]$')
        self.gen_unknown_lexeme1_regex = re.compile(pattern=r'^WARNING: unknown lexeme \'be_inv_am\'!$')
        self.gen_unknown_lexeme2_regex = re.compile(pattern=r'^ERROR: trigger rules call for non-existant lexeme `be_inv_am\'$')
        self.gen_final0_regex = re.compile(pattern=r'^NOTE: transfer did 0 successful unifies and 0 failed ones$')
        self.gen_final1_regex = re.compile(pattern=r'^NOTE: generated [0-9]+ / [1-9][0-9]* sentences, avg [0-9]+k, time [0-9]+.[0-9]+s$')
        self.gen_final2_regex = re.compile(pattern=r'^NOTE: transfer did [0-9]+ successful unifies and [0-9]+ failed ones$')

    def parse(self, sentence=None, sentence_list=None, print_notes=False):
        assert (sentence is None) != (sentence_list is None)
        if sentence is not None:
            sentence_list = [sentence]

        parses_iter = self.parse_iter(sentence_list=sentence_list, print_notes=print_notes)

        if sentence is None:
            return parses_iter
        else:
            parses = next(parses_iter)
            try:
                next(parses_iter)
                assert False
            except StopIteration:
                pass
            return parses

    def parse_iter(self, sentence_list=None, print_notes=False):
        sentence_list = list(sentence_list)
        num_sentences = len(sentence_list)
        ace_input = '\n'.join(sentence_list) + '\n'

        ace_parse = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = ace_parse.communicate(input=ace_input.encode())

        # stderr
        lines = iter(stderr_data.decode('utf-8').splitlines())
        num_results = list()
        while True:
            line = next(lines)
            if self.parse_successful_regex.match(line):
                num_results.append(int(line[6: line.index(' readings, added')]))
                continue
            elif self.parse_unsuccessful_regex.match(line):
                num_results.append(0)
                continue
            elif self.parse_final0_regex.match(line):
                assert len(sentence_list) == 0
                raise Exception('Empty sentence list!')
            elif self.parse_final1_regex.match(line):
                assert len(num_results) == num_sentences
                break
            else:
                assert False, line

        # stdout
        lines = iter(stdout_data.decode('utf-8').splitlines())

        for n, count in enumerate(num_results):
            line = next(lines)

            if count == 0:
                assert line == ('SKIP: ' + sentence_list[n])
            else:
                assert line == ('SENT: ' + sentence_list[n])

            parses = [next(lines) for _ in range(count)]  # list actually performs next calls
            parses = (self.parse_item(line=line, print_notes=print_notes) for line in parses)  # do not parse items immediately

            line = next(lines)
            assert line == ''
            line = next(lines)
            assert line == ''

            if count == 0:
                yield None
            elif not self.first_only:
                yield parses
            else:
                parse = next(parses)
                try:
                    next(parses)
                    assert False
                except StopIteration:
                    pass
                yield parse

        try:
            next(lines)
            assert False
        except StopIteration:
            pass

    def parse_item(self, line, print_notes=False):
        try:
            if self.trees:
                mrs = Mrs.from_string(line[:line.index(' ; ')])
                tree = DerivTree.from_string(line[line.index(' ; ') + 4:])
                return mrs, tree
            else:
                mrs = Mrs.from_string(line)
                return mrs
        except Exception as exc:
            if print_notes:
                print(exc)
            if str(exc).startswith('Label ') and ' of hcon must be associated with an EP.' in str(exc):
                return None
            elif str(exc).startswith('Variable ') and ' is already defined.' in str(exc):
                return None
            elif str(exc).startswith('Heuristics failed: '):
                return None
            elif str(exc).startswith('Problem4: '):
                return None
            elif str(exc).startswith('Ambiguous head heuristic'):
                return None
            elif str(exc) == 'Invalid ElemPred.':
                return None
            else:
                raise exc

    def generate(self, mrs=None, mrs_list=None, print_notes=False):
        assert (mrs is None) != (mrs_list is None)
        if mrs is not None:
            mrs_list = [mrs]

        generated_iter = self.generate_iter(mrs_list=mrs_list, print_notes=print_notes)

        if mrs is None:
            return generated_iter
        else:
            generated = next(generated_iter)
            try:
                next(generated_iter)
                assert False
            except StopIteration:
                pass
            return generated

    def generate_iter(self, mrs=None, mrs_list=None, print_notes=False):
        num_mrs = len(mrs_list)
        ace_input = '\n'.join(str(mrs) for mrs in mrs_list) + '\n'

        ace_generate = subprocess.Popen(self.args + ['-e'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = ace_generate.communicate(input=ace_input.encode())

        # stderr
        lines = iter(stderr_data.decode('utf-8').splitlines())
        num_results = list()
        successful = None
        while True:
            line = next(lines)
            if self.gen_successful_regex.match(line):
                num_results.append(int(line[line.rindex('[') + 1: -9]))
                successful = True
                continue
            elif self.gen_unsuccessful_regex.match(line):
                num_results.append(0)
                successful = False
                continue
            elif self.gen_unknown_lexeme1_regex.match(line):
                assert successful is not None
                if print_notes:
                    sys.stderr.write('{} {}\n'.format(len(num_results) - 1, line))
                line = next(lines)
                self.gen_unknown_lexeme2_regex.match(line)
                if print_notes:
                    sys.stderr.write('{} {}\n'.format(len(num_results) - 1, line))
                successful = None
            elif self.gen_final0_regex.match(line):
                assert len(mrs_list) == 0
                raise Exception('Empty MRS list!')
            elif self.gen_final1_regex.match(line):
                assert len(num_results) == num_mrs
                break
            else:
                assert False, line
        self.gen_final2_regex.match(next(lines))
        try:
            next(lines)
            assert False
        except StopIteration:
            pass

        # stdout
        lines = iter(stdout_data.decode('utf-8').splitlines())

        for count in num_results:
            generated = list()
            for _ in range(count):
                sentence = next(lines)
                generated.append(sentence)
            line = next(lines)
            assert line == ''

            if not self.first_only:
                yield generated
            elif count > 0:
                assert len(generated) == 1
                yield generated[0]
            else:
                yield None

        try:
            next(lines)
            assert False
        except StopIteration:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ACE')
    parser.add_argument('-x', '--executable', default='/home/aok25/ShapeWorld/shapeworld/realizers/dmrs/resources/ace', help='Executable')
    parser.add_argument('-g', '--grammar', default='/home/aok25/ShapeWorld/shapeworld/realizers/dmrs/languages/english.dat', help='Grammar')
    parser.add_argument('-e', '--generate', action='store_true', help='Generation mode')
    parser.add_argument('-n', '--num-outputs', type=int, default=1, help='Number of outputs')
    parser.add_argument('-t', '--trees', action='store_true', help='Derivation trees')
    parser.add_argument('-d', '--dmrs', action='store_true', help='DMRS graph')
    parser.add_argument('-r', '--root', help='Root')
    parser.add_argument('-i', '--informal', action='store_true', help='root_informal')
    args = parser.parse_args()

    ace = Ace(executable=args.executable, grammar=args.grammar, num_outputs=args.num_outputs, trees=args.trees, root=args.root, informal=args.informal)

    if sys.stdin.isatty():

        if args.generate:
            while True:
                if args.dmrs:
                    sys.stdout.write('DMRS: ')
                    sys.stdout.flush()
                    dmrs = sys.stdin.readline().strip()
                    dmrs = ListDmrs.loads_xml(bytestring=dmrs)
                    mrs = dmrs.convert_to(Mrs)
                else:
                    sys.stdout.write('MRS: ')
                    sys.stdout.flush()
                    mrs = sys.stdin.readline().strip()
                generated = ace.generate(mrs=mrs, print_notes=False)
                if args.num_outputs == 1:
                    generated = [generated]
                sys.stdout.write('\n'.join('{n} {sent}'.format(n=n, sent=sentence) for n, sentence in enumerate(generated)) + '\n\n')
                sys.stdout.flush()

        else:
            while True:
                sys.stdout.write('SENTENCE: ')
                sys.stdout.flush()
                sentence = sys.stdin.readline().strip()
                parse = ace.parse(sentence=sentence, print_notes=False)
                if parse is None:
                    sys.stdout.write('Sentence could not be parsed.\n\n')
                elif args.num_outputs == 1:
                    if args.dmrs:
                        assert not args.trees
                        sys.stdout.write('MRS: {mrs}\nDMRS: {dmrs}\n\n'.format(mrs=parse, dmrs=parse.convert_to(ListDmrs).dumps_xml().decode()))
                    elif args.trees:
                        sys.stdout.write('MRS: {mrs}\nDERIV: {deriv}\n\n'.format(mrs=parse[0], deriv=parse[1]))
                    else:
                        sys.stdout.write('{mrs}\n\n'.format(mrs=parse))
                else:
                    if args.dmrs:
                        assert not args.trees
                        sys.stdout.write('\n\n'.join('{n}\nMRS: {mrs}\nDMRS: {dmrs}'.format(n=n, mrs=mrs, dmrs=(None if mrs is None else mrs.convert_to(ListDmrs).dumps_xml().decode())) for n, mrs in enumerate(parse)) + '\n\n')
                    elif args.trees:
                        sys.stdout.write('\n\n'.join('{n}\nMRS: {mrs}\nDERIV: {deriv}'.format(n=n, mrs=mrs, deriv=deriv) for n, (mrs, deriv) in enumerate(parse)) + '\n\n')
                    else:
                        sys.stdout.write('\n\n'.join('{n}: {mrs}'.format(n=n, mrs=mrs) for n, mrs in enumerate(parse)) + '\n\n')
                sys.stdout.flush()

    else:

        if args.generate:
            generated_iter = ace.generate(mrs_list=[line.strip() for line in sys.stdin], print_notes=False)
            sys.stdout.write('\n\n'.join('\n'.join(generated) for generated in generated_iter) + '\n\n')

        else:
            parses_iter = ace.parse(sentence_list=[line.strip() for line in sys.stdin], print_notes=False)
            if args.trees:
                sys.stdout.write('\n\n'.join(str(parses[0]) + '\n' + str(parses[1]) if args.num_outputs == 1 else '\n'.join(str(mrs) + '\n' + str(deriv) for mrs, deriv in parses) for parses in parses_iter) + '\n\n')
            else:
                sys.stdout.write('\n\n'.join(str(parses) if args.num_outputs == 1 else '\n'.join(str(mrs) for mrs in parses) for parses in parses_iter) + '\n\n')
