import argparse
from datetime import datetime
import json
import os
import shutil
import sys
from shapeworld import dataset, util


def parse_tuple(string):
    assert string
    if string[0] == '(' and string[-1] == ')':
        string = string[1:-1]
    return tuple(util.parse_int_with_factor(x) for x in string.split(','))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate example data')

    parser.add_argument('-d', '--directory', help='Directory for generated data ( with automatically created sub-directories)')
    parser.add_argument('-u', '--directory-unmanaged', default=None, help='Directory for generated data (without automatically created sub-directories)')
    parser.add_argument('-a', '--archive', default=None, choices=('zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma'), help='Store generated data in (compressed) archive')
    parser.add_argument('-A', '--append', action='store_true', help='Append to existing data (when used with --directory)')

    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
    parser.add_argument('-c', '--config', type=util.parse_config, default=None, help='Dataset configuration file')

    parser.add_argument('-p', '--parts', type=parse_tuple, default=None, help='Number of parts')
    parser.add_argument('-i', '--instances', type=util.parse_int_with_factor, default=100, help='Number of instances (per part)')
    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test'), help='Mode')

    parser.add_argument('-M', '--include-model', action='store_true', help='Include world/caption model')
    parser.add_argument('-P', '--no-pixel-noise', action='store_true', help='Do not infuse pixel noise')
    parser.add_argument('-C', '--concatenate-images', action='store_true', help='Concatenate images per part in one image')
    parser.add_argument('-S', '--captioner-statistics', action='store_true', help='Collect statistical data of captioner')
    # parser.add_argument('-v', '--values', default=None, help='Comma-separated list of values to include')
    args = parser.parse_args()

    dataset = dataset(dtype=args.type, name=args.name, config=args.config)
    sys.stdout.write('{time} {dtype} dataset: {name}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=dataset.type, name=dataset.name))
    sys.stdout.write('         config: {config}\n'.format(config=args.config))
    sys.stdout.flush()

    if args.instances * dataset.world_size * dataset.world_size * 3 > 5e8:  # > 500MB
        sys.stdout.write('{time} Warning: part size is {size}MB\n'.format(time=datetime.now().strftime('%H:%M:%S'), size=int(args.instances * dataset.world_size * dataset.world_size * 3 / 1e6)))
        sys.stdout.flush()
        if sys.stdin.readline()[:-1].lower() in ('n', 'no', 'c', 'cancel', 'abort'):
            exit(0)

    specification = dataset.specification()
    if args.archive:
        specification['archive'] = args.archive
    if args.include_model:
        specification['include_model'] = args.include_model
    if args.no_pixel_noise:
        specification['noise_range'] = dataset.world_generator.noise_range
    if args.concatenate_images:
        specification['num_concat_worlds'] = args.instances

    if args.directory:
        assert not args.directory_unmanaged
        assert not args.mode
        assert not args.parts or len(args.parts) in (1, 3, 4)

        directory = os.path.join(args.directory, dataset.type, dataset.name)
        specification_path = os.path.join(args.directory, '{}-{}.json'.format(dataset.type, dataset.name))
        parts = args.parts or (1, 1, 1)

        if len(parts) == 3:
            modes = ('train', 'validation', 'test')
            directories = tuple(os.path.join(directory, mode) for mode in modes)
            tf_records_flags = (False, False, False)
        elif len(parts) == 1:
            from shapeworld import tf_util
            modes = ('train',)
            directories = (os.path.join(directory, 'tf-records'),)
            tf_records_flags = (True,)
        else:
            from shapeworld import tf_util
            modes = ('train', 'train', 'validation', 'test')
            directories = tuple(os.path.join(directory, mode) for mode in ('tf-records', 'train', 'validation', 'test'))
            tf_records_flags = (True, False, False, False)

        if args.append:
            start_part = ()
            for subdir in directories:
                for root, dirs, files in os.walk(subdir):
                    if root == subdir:
                        if 'captioner_statistics.csv' in files:
                            files.remove('captioner_statistics.csv')
                        if dirs:
                            assert all(d[:4] == 'part' for d in dirs)
                            start_part += (max(int(d[4:]) for d in dirs),)
                        elif files:
                            assert all(f[:4] == 'part' for f in files)
                            start_part += (max(int(f[4:f.index('.')]) for f in files),)
                        else:
                            start_part += (0,)
            with open(specification_path, 'r') as filehandle:
                assert json.load(filehandle) == specification
        else:
            start_part = (0,) * len(directories)
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            with open(specification_path, 'w') as filehandle:
                filehandle.write(json.dumps(specification))
            if len(directories) > 1:
                for subdir in directories:
                    os.mkdir(subdir)

    elif args.directory_unmanaged:
        assert not args.parts or len(args.parts) == 1
        directory = args.directory_unmanaged
        modes = (args.mode,)
        directories = (args.directory_unmanaged,)
        parts = args.parts or (1,)

    else:
        assert False

    if args.directory_unmanaged and len(parts) == 1 and parts[0] == 1:
        if args.captioner_statistics:
            dataset.collect_captioner_statistics(path=os.path.join(directories[0], 'captioner_statistics.csv'), append=args.append)
        sys.stdout.write('{time} generate {dtype} {name}{mode} data...\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=dataset.type, name=dataset.name, mode=(' ' + modes[0] if modes[0] else '')))
        sys.stdout.flush()
        generated = dataset.generate(n=args.instances, mode=modes[0], noise=(not args.no_pixel_noise), include_model=args.include_model)
        dataset.serialize(path=directories[0], generated=generated, archive=args.archive, concat_worlds=args.concatenate_images)
        if args.captioner_statistics:
            dataset.close_captioner_statistics()
        sys.stdout.write('{time} data generation completed!\n'.format(time=datetime.now().strftime('%H:%M:%S')))
        sys.stdout.flush()
    else:
        for mode, directory, num_parts, start, tf_records_flag in zip(modes, directories, parts, start_part, tf_records_flags):
            if args.captioner_statistics:
                dataset.collect_captioner_statistics(path=os.path.join(directory, 'captioner_statistics.csv'), append=args.append)
            sys.stdout.write('{time} generate {dtype} {name}{mode} data...\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=dataset.type, name=dataset.name, mode=(' ' + mode if mode else '')))
            sys.stdout.write('         0%  0/{parts}  (time per part: n/a)'.format(parts=num_parts))
            sys.stdout.flush()
            for part in range(1, num_parts + 1):
                before = datetime.now()
                generated = dataset.generate(n=args.instances, mode=mode, noise=(not args.no_pixel_noise), include_model=args.include_model)
                if tf_records_flag:
                    tf_util.write_records(path=os.path.join(directory, 'part{}'.format(start + part)), records=generated, dataset=dataset)
                else:
                    dataset.serialize(path=os.path.join(directory, 'part{}'.format(start + part)), generated=generated, archive=args.archive, concat_worlds=args.concatenate_images)
                after = datetime.now()
                sys.stdout.write('\r         {completed:.0f}%  {part}/{parts}  (time per part: {duration})'.format(completed=((part) * 100 / num_parts), part=part, parts=num_parts, duration=str(after - before).split('.')[0]))
                sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.flush()
            if args.captioner_statistics:
                dataset.close_captioner_statistics()
        sys.stdout.write('{time} data generation completed!\n'.format(time=datetime.now().strftime('%H:%M:%S')))
        sys.stdout.flush()
