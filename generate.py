import argparse
from datetime import datetime
import json
import os
import shutil
import sys
from shapeworld import Dataset, util


def parse_triple(string):
    assert string
    n = string.count(',')
    assert n == 0 or n == 2
    if string[0] == '(' and string[-1] == ')':
        string = string[1:-1]
        assert n == 2
    return tuple(util.parse_int_with_factor(x) for x in string.split(','))


def parse_config(string):
    assert string
    if string[0] == '{':
        if '\'' in string and '\"' not in string:
            string = string.replace('\'', '\"')
        return json.loads(string)
    else:
        return string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate example data')

    parser.add_argument('-d', '--directory', help='Directory for generated data ( with automatically created sub-directories)')
    parser.add_argument('-u', '--directory-unmanaged', default=None, help='Directory for generated data (without automatically created sub-directories)')
    parser.add_argument('-a', '--archive', default=None, choices=('zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma'), help='Store generated data in (compressed) archive')
    parser.add_argument('-A', '--append', action='store_true', help='Append to existing data (when used with --directory)')

    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
    parser.add_argument('-c', '--config', type=parse_config, default=None, help='Dataset configuration file')

    parser.add_argument('-p', '--parts', type=parse_triple, default=None, help='Number of parts')
    parser.add_argument('-i', '--instances', type=util.parse_int_with_factor, default=100, help='Number of instances (per part)')
    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test'), help='Mode')

    parser.add_argument('-W', '--world-model', action='store_true', help='Include world model')
    parser.add_argument('-P', '--no-pixel-noise', action='store_true', help='Do not infuse pixel noise')
    parser.add_argument('-S', '--captioner-statistics', action='store_true', help='Collect statistical data of captioner')
    # parser.add_argument('-v', '--values', default=None, help='Comma-separated list of values to include')
    parser.add_argument('-T', '--tiff', action='store_true', help='Store images in tiff format using LZW compression')
    args = parser.parse_args()

    dataset = Dataset.from_config(config=args.config, dataset_type=args.type, dataset_name=args.name)

    if args.directory:
        assert not args.directory_unmanaged
        assert not args.mode
        assert not args.parts or len(args.parts) == 3

        specification = dataset.specification()
        specification['world_model'] = args.world_model
        if args.archive:
            specification['archive'] = args.archive
        if args.no_pixel_noise:
            specification['noise_range'] = dataset.world_generator.noise_range
        if args.tiff:
            specification['tiff'] = args.tiff

        directory = os.path.join(args.directory, dataset.type, dataset.name)
        modes = ('train', 'validation', 'test')
        directories = tuple(os.path.join(directory, mode) for mode in modes)
        parts = args.parts or (1, 1, 1)

        if args.append:
            start_part = ()
            for directory in directories:
                for root, dirs, files in os.walk(directory):
                    if root == directory:
                        assert all(d[:4] == 'part' for d in dirs)
                        start_part += (max(int(d[4:]) for d in dirs) + 1,)
            with open(os.path.join(directory, 'specification.json'), 'r') as filehandle:
                assert json.load(filehandle) == specification
        else:
            start_part = (0, 0, 0)
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            with open(os.path.join(directory, 'specification.json'), 'w') as filehandle:
                filehandle.write(json.dumps(specification))
            for directory in directories:
                os.mkdir(directory)

    elif args.directory_unmanaged:
        assert not args.append
        assert not args.parts or len(args.parts) == 1
        modes = (args.mode,)
        directories = (args.directory_unmanaged,)
        parts = args.parts or (1,)
        start_part = (0,)

    else:
        assert False

    if len(parts) == 1 and parts[0] == 1:
        sys.stdout.write('{} Generate {}{} data...\n'.format(datetime.now().strftime('%H:%M:%S'), dataset, ' ' + modes[0] if modes[0] else ''))
        sys.stdout.flush()
        generated = dataset.generate(n=args.instances, mode=modes[0], noise=(not args.no_pixel_noise), include_model=args.world_model)
        dataset.serialize_data(directory=directories[0], generated=generated, archive=args.archive, tiff=args.tiff)
        sys.stdout.write('{} Data generation completed!\n'.format(datetime.now().strftime('%H:%M:%S')))
        sys.stdout.flush()
    else:
        for mode, directory, num_parts, start in zip(modes, directories, parts, start_part):
            if args.captioner_statistics:
                filehandle = open(os.path.join(directory, 'captioner_statistics.csv'), 'a' if args.append else 'w')
                dataset.collect_captioner_statistics(filehandle=filehandle, append=args.append)
            sys.stdout.write('{} Generate {}{} data...\n'.format(datetime.now().strftime('%H:%M:%S'), dataset, ' ' + mode if mode else ''))
            sys.stdout.write('         0%  0/{}  (time per part: n/a)'.format(num_parts))
            sys.stdout.flush()
            for part in range(num_parts):
                before = datetime.now()
                generated = dataset.generate(n=args.instances, mode=mode, noise=(not args.no_pixel_noise), include_model=args.world_model)
                dataset.serialize_data(directory=directory, generated=generated, name='part{}'.format(start + part), archive=args.archive, tiff=args.tiff)
                after = datetime.now()
                sys.stdout.write('\r         {:.0f}%  {}/{}  (time per part: {})'.format((part + 1) * 100 / num_parts, part + 1, num_parts, str(after - before).split('.')[0]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
            if args.captioner_statistics:
                dataset.close_captioner_statistics()
        sys.stdout.write('{} Data generation completed!\n'.format(datetime.now().strftime('%H:%M:%S')))
        sys.stdout.flush()
