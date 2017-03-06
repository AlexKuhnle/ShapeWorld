import argparse
from datetime import datetime
import json
import os
import shutil
from shapeworld.dataset import Dataset


def triple(string):
    n = string.count(',')
    assert n == 0 or n == 2
    if string[0] == '(' and string[-1] == ')':
        string = string[1:-1]
        assert n == 2
    if n == 2:
        return tuple(int(x) for x in string.split(','))
    else:
        return (int(string),)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate example data')
    parser.add_argument('-d', '--directory', default='examples', help='Directory for generated data')
    parser.add_argument('-a', '--archive', default=None, choices=('zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma'), help='Store generated data in archive')
    parser.add_argument('-s', '--specification', default=None, help='Specification file')
    parser.add_argument('-t', '--type', default='agreement', help='Dataset type')
    parser.add_argument('-n', '--name', default='oneshape', help='Dataset name')
    parser.add_argument('-c', '--config', default=None, help='Dataset configuration file')
    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test'), help='Mode')
    parser.add_argument('-i', '--instances', type=int, default=100, help='Number of instances (per batch)')
    parser.add_argument('-b', '--batches', type=triple, default=(1,), help='Number of batches')
    parser.add_argument('-w', '--world-model', action='store_true', help='Include world model')
    parser.add_argument('-p', '--pixel-noise-off', action='store_true', help='Turn pixel noise off')
    # parser.add_argument('-v', '--values', default=None, help='Comma-separated list of values to include')
    parser.add_argument('-f', '--tiff', action='store_true', help='Store images in tiff format using LZW compression')
    args = parser.parse_args()

    if os.path.isdir(args.directory):
        shutil.rmtree(args.directory)
    os.makedirs(args.directory)

    dataset = Dataset.from_config(config=args.config, dataset_type=args.type, dataset_name=args.name)

    if len(args.batches) > 1:
        assert len(args.batches) == 3 and not args.mode
        modes = ['train', 'validation', 'test']
        directories = [os.path.join(args.directory, mode) for mode in modes]
    else:
        modes = [args.mode]
        directories = [args.directory]
    for mode, directory, num_batches in zip(modes, directories, args.batches):
        print('{} {}: '.format(datetime.now().strftime('%H:%M:%S'), mode or 'instances'), end='', flush=True)
        num_batches = int(num_batches)
        if num_batches > 1:
            for n in range(num_batches):
                if (n + 1) % max(num_batches // 20, 1) == 0:
                    print('.', end='', flush=True)
                generated = dataset.generate(n=args.instances, mode=mode, noise=(not args.pixel_noise_off), include_model=args.world_model)
                dataset.serialize_data(directory=os.path.join(directory, 'batch{}'.format(n)), generated=generated, archive=args.archive, tiff=args.tiff)
        else:
            generated = dataset.generate(n=args.instances, mode=mode, noise=(not args.pixel_noise_off), include_model=args.world_model)
            dataset.serialize_data(directory=directory, generated=generated, archive=args.archive, tiff=args.tiff)
        print(' done', flush=True)

    specification = dataset.specification()
    specification['directory'] = args.directory
    specification['world_model'] = args.world_model
    specification['tiff'] = args.tiff
    specification['noise_range'] = dataset.world_generator.noise_range if args.pixel_noise_off else None
    specification['archive'] = args.archive
    with open(os.path.join(args.directory, 'specification.json'), 'w') as filehandle:
        filehandle.write(json.dumps(specification))
    if args.specification:
        with open(args.specification, 'w') as filehandle:
            filehandle.write(json.dumps(specification))
