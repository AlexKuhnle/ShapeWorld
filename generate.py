import argparse
from datetime import datetime
import json
import os
import shutil
import sys
from shapeworld import dataset, util


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate example data')

    parser.add_argument('-d', '--directory', help='Directory for generated data (with automatically created sub-directories, unless --unmanaged)')
    parser.add_argument('-a', '--archive', default=None, choices=('zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma'), help='Store generated data in (compressed) archives')
    parser.add_argument('-A', '--append', action='store_true', help='Append to existing data')
    parser.add_argument('-U', '--unmanaged', action='store_true', help='Do not automatically create sub-directories (implied if --files not specified)')

    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', help='Dataset name')
    parser.add_argument('-l', '--language', default=None, help='Dataset language')
    parser.add_argument('-c', '--config', type=util.parse_config, default=None, help='Dataset configuration file')

    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test', 'tf-records'), help='Mode')
    parser.add_argument('-f', '--files', type=util.parse_tuple, default=None, help='Number of files to split data into (not specified implies --unmanaged)')
    parser.add_argument('-i', '--instances', type=util.parse_int_with_factor, default=100, help='Number of instances per file')

    parser.add_argument('-p', '--pixel-noise', type=float, default=0.0, help='Pixel noise range')
    parser.add_argument('-M', '--include-model', action='store_true', help='Include world/caption model (as json file)')
    parser.add_argument('-C', '--concatenate-images', action='store_true', help='Concatenate images per part into one image file')
    parser.add_argument('-H', '--html', action='store_true', help='Create HTML file showing the generated data')
    # parser.add_argument('-v', '--values', default=None, help='Comma-separated list of values to include')
    args = parser.parse_args()

    dataset = dataset(dtype=args.type, name=args.name, language=args.language, config=args.config)
    sys.stdout.write('{time} {dataset}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dataset=dataset))
    sys.stdout.write('         config: {config}\n'.format(config=args.config))
    sys.stdout.flush()

    if args.instances * util.product(dataset.world_shape) > 5e8:  # > 500MB
        sys.stdout.write('{time} warning: part size is {size}MB '.format(time=datetime.now().strftime('%H:%M:%S'), size=int(args.instances * util.product(dataset.world_shape) / 1e6)))
        sys.stdout.flush()
        if sys.stdin.readline()[:-1].lower() in ('n', 'no', 'c', 'cancel', 'a', 'abort'):
            exit(0)

    specification = dataset.specification()
    if args.archive:
        specification['archive'] = args.archive
    if args.include_model:
        specification['include_model'] = args.include_model
    if args.concatenate_images:
        specification['num_concat_worlds'] = args.instances

    args.unmanaged = args.unmanaged or (args.files is None)
    if args.unmanaged:
        directory = args.directory
        directories = (args.directory,)
        if args.files is None:
            parts = (1,)
        else:
            assert len(args.files) == 1
            parts = args.files
        if args.mode == 'tf-records':
            from shapeworld import tf_util
            modes = ('train',)
            tf_records_flags = (True,)
        else:
            modes = (args.mode,)
            tf_records_flags = (False,)

    else:
        assert len(args.files) in (1, 3, 4)
        assert (args.mode is not None) == (len(args.files) == 1)
        if dataset.language is None:
            directory = os.path.join(args.directory, dataset.type, dataset.name)
            specification_path = os.path.join(args.directory, '{}-{}.json'.format(dataset.type, dataset.name))
        else:
            directory = os.path.join(args.directory, '{}-{}'.format(dataset.type, dataset.language), dataset.name)
            specification_path = os.path.join(args.directory, '{}-{}-{}.json'.format(dataset.type, dataset.language, dataset.name))

        parts = args.files
        if len(parts) == 1:
            if args.mode == 'tf-records':
                from shapeworld import tf_util
                modes = ('train',)
                tf_records_flags = (True,)
            else:
                modes = (args.mode,)
                tf_records_flags = (False,)
            directories = (os.path.join(directory, args.mode),)
        elif len(parts) == 3:
            assert args.mode is None
            modes = ('train', 'validation', 'test')
            directories = tuple(os.path.join(directory, mode) for mode in modes)
            tf_records_flags = (False, False, False)
        else:
            assert args.mode is None
            from shapeworld import tf_util
            modes = ('train', 'train', 'validation', 'test')
            directories = tuple(os.path.join(directory, mode) for mode in ('tf-records', 'train', 'validation', 'test'))
            tf_records_flags = (True, False, False, False)

    if args.append:
        start_part = ()
        for subdir in directories:
            for root, dirs, files in os.walk(subdir):
                if root == subdir:
                    if dirs:
                        assert all(d[:4] == 'part' for d in dirs)
                        start_part += (max(int(d[4:]) for d in dirs),)
                    elif files:
                        assert all(f[:4] == 'part' for f in files)
                        start_part += (max(int(f[4:f.index('.')]) for f in files),)
                    else:
                        start_part += (0,)
        if not args.unmanaged:
            with open(specification_path, 'r') as filehandle:
                assert json.load(filehandle) == specification, str(specification)
    else:
        start_part = (0,) * len(directories)
        if not args.unmanaged and os.path.isdir(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        elif not os.path.isdir(directory):
            os.makedirs(directory)
        if not args.unmanaged:
            with open(specification_path, 'w') as filehandle:
                filehandle.write(json.dumps(specification))
        if len(directories) > 1:
            for subdir in directories:
                os.makedirs(subdir)

    for mode, directory, num_parts, start, tf_records_flag in zip(modes, directories, parts, start_part, tf_records_flags):
        sys.stdout.write('{time} generate {dtype} {name}{mode} data...\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=dataset.type, name=dataset.name, mode=(' ' + mode if mode else '')))
        sys.stdout.write('         0%  0/{files}  (time per part: n/a)'.format(files=num_parts))
        sys.stdout.flush()
        for part in range(1, num_parts + 1):
            before = datetime.now()
            if args.unmanaged and len(parts) == 1 and parts[0] == 1:
                path = directory
            else:
                path = os.path.join(directory, 'part{}'.format(start + part))
            generated = dataset.generate(n=args.instances, mode=mode, noise_range=args.pixel_noise, include_model=args.include_model, alternatives=True)
            if generated is None:
                assert False
            elif tf_records_flag:
                tf_util.write_records(dataset=dataset, records=generated, path=path)
            else:
                dataset.serialize(path=path, generated=generated, archive=args.archive, concat_worlds=args.concatenate_images, html=args.html)
            after = datetime.now()
            sys.stdout.write('\r         {completed:.0f}%  {part}/{parts}  (time per part: {duration})'.format(completed=((part) * 100 / num_parts), part=part, parts=num_parts, duration=str(after - before).split('.')[0]))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    sys.stdout.write('{time} data generation completed\n'.format(time=datetime.now().strftime('%H:%M:%S')))
    sys.stdout.flush()
