import argparse
from datetime import datetime
from io import BytesIO
import json
import os
import shutil
import sys
from shapeworld import Dataset, util


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate example data')

    parser.add_argument('-d', '--directory', help='Directory for generated data (with automatically created sub-directories, unless --unmanaged)')
    parser.add_argument('-a', '--archive', default=None, choices=('zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma'), help='Store generated data in (compressed) archives')
    parser.add_argument('-U', '--unmanaged', action='store_true', help='Do not automatically create sub-directories (implied if --shards not specified)')

    parser.add_argument('-t', '--type', default=None, help='Dataset type')
    parser.add_argument('-n', '--name', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Dataset name')
    parser.add_argument('-v', '--variant', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Label for configuration variant')
    parser.add_argument('-l', '--language', default=None, help='Language')
    parser.add_argument('-c', '--config', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Configuration file/directory')

    parser.add_argument('-s', '--shards', type=util.parse_tuple(parse_item=util.parse_int_with_factor, unary_tuple=True, valid_sizes=(1, 3)), default=None, help='Number of shards to split data into (not specified implies --unmanaged)')
    parser.add_argument('-i', '--instances', type=util.parse_int_with_factor, default=128, help='Number of instances per shard')
    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test'), help='Mode')
    parser.add_argument('-A', '--append', action='store_true', help='Append to existing data (when used without --unmanaged)')
    parser.add_argument('-b', '--begin', type=util.parse_tuple(parse_item=util.parse_int_with_factor, unary_tuple=True, valid_sizes=(1, 3)), default=None, help='Begin from shard number (requires --append)')

    parser.add_argument('-P', '--delay-pixel-noise', action='store_true', help='Do not infuse pixel noise now, but when dataset is loaded')
    parser.add_argument('-M', '--include-model', action='store_true', help='Include world/caption model (as json file)')
    parser.add_argument('-H', '--html', action='store_true', help='Create HTML file visualizing the generated data')
    parser.add_argument('-T', '--tf-records', action='store_true', help='Additionally store data as TensorFlow records')
    parser.add_argument('-F', '--features', action='store_true', help='Additionally extract image features (conv4 of resnet_v2_101)')
    parser.add_argument('-C', '--clevr-format', action='store_true', help='Output in CLEVR format')
    parser.add_argument('-N', '--png-format', action='store_true', help='Store images in PNG as opposed to bitmap format')
    parser.add_argument('-O', '--concatenate-images', action='store_true', help='Concatenate images per part into one image file')

    parser.add_argument('-Y', '--yes', action='store_true', help='Confirm all questions with yes')
    # parser.add_argument('-v', '--values', default=None, help='Comma-separated list of values to include')
    parser.add_argument('--v1', action='store_true')

    parser.add_argument('--config-values', nargs=argparse.REMAINDER, default=(), help='Additional dataset configuration values passed as command line arguments')

    args = parser.parse_args()
    args.config_values = util.parse_config(values=args.config_values)

    # TFRecords utility
    if args.tf_records:
        from shapeworld import tf_util

    if args.v1:
        util.set_version(1)

    # does not include variant, as loading data for generation is not expected
    dataset = Dataset.create(dtype=args.type, name=args.name, language=args.language, config=args.config, **args.config_values)
    sys.stdout.write('{time} {dataset}\n'.format(time=datetime.now().strftime('%H:%M:%S'), dataset=dataset))
    if args.config is None:
        if args.config_values:
            sys.stdout.write('         config: {config}\n'.format(config=args.config_values))
    else:
        sys.stdout.write('         config: {config}\n'.format(config=args.config))
        if args.config_values:
            sys.stdout.write('                 {config}\n'.format(config=args.config_values))
    sys.stdout.flush()

    if args.archive is not None and not args.delay_pixel_noise and dataset.pixel_noise_stddev > 0.0:
        sys.stdout.write('Warning: best compression results without pixel noise, continue? ')
        sys.stdout.flush()
        if args.yes:
            sys.stdout.write('y\n')
        elif util.negative_response(sys.stdin.readline()[:-1]):
            exit(0)

    if args.instances * util.product(dataset.world_shape()) > 5e8:  # > 500MB
        sys.stdout.write('{time} warning: shard size is {size}MB '.format(time=datetime.now().strftime('%H:%M:%S'), size=int(args.instances * util.product(dataset.world_shape()) / 1e6)))
        sys.stdout.flush()
        if args.yes:
            sys.stdout.write('y\n')
        elif util.negative_response(sys.stdin.readline()[:-1]):
            exit(0)

    if args.features:
        from pretrained import PretrainedModel
        pretrained_model = PretrainedModel(image_shape=dataset.world_shape())
        for value_name, value_type in list(dataset.values.items()):
            value_type, alts = util.alternatives_type(value_type=value_type)
            if value_type == 'world':
                if alts:
                    dataset.values[value_name + '_features'] = 'alternatives(vector(float))'
                else:
                    dataset.values[value_name + '_features'] = 'vector(float)'
                dataset.vectors[value_name + '_features'] = pretrained_model.features_shape

    specification = dataset.specification()
    if args.archive:
        specification['archive'] = args.archive
    if args.delay_pixel_noise and dataset.pixel_noise_stddev > 0.0:
        specification['pixel_noise_stddev'] = dataset.pixel_noise_stddev
        dataset.pixel_noise_stddev = 0.0
    if args.include_model:
        specification['include_model'] = args.include_model
    if args.png_format:
        specification['image_format'] = 'png'
    if args.concatenate_images:
        specification['num_concat_worlds'] = args.instances

    args.unmanaged = args.unmanaged or (args.shards is None)
    if args.unmanaged:
        directory = args.directory
        if args.shards is None:
            shards = (0,)
        else:
            shards = args.shards

    else:
        full_name = dataset.name
        if args.variant is not None:
            full_name = '{}-{}'.format(full_name, args.variant)
        if args.language is not None:
            full_name = '{}-{}'.format(full_name, args.language)
        directory = os.path.join(args.directory, dataset.type, full_name)
        specification_path = os.path.join(args.directory, '{}-{}.json'.format(dataset.type, full_name))
        shards = args.shards

    assert all(shard >= 0 for shard in shards)
    assert args.shards is None or (args.mode is not None) == (len(shards) == 1)
    if len(shards) == 1:
        modes = (args.mode,)
        if args.unmanaged:
            directories = (directory,)
        else:
            directories = (os.path.join(directory, args.mode),)
    elif len(shards) == 3:
        assert args.mode is None
        modes = ('train', 'validation', 'test')
        directories = tuple(os.path.join(directory, mode) for mode in modes)
    if all(shard == 0 for shard in shards):
        shards = tuple(None for _ in shards)

    assert args.begin is None or args.append
    assert args.begin is None or len(args.begin) == len(args.shards)
    if args.append:
        if args.begin is not None:
            shards_begin = args.begin
        elif args.clevr_format:
            shards_begin = tuple(0 for _ in directories)
        else:
            shards_begin = ()
            for subdir in directories:
                for root, dirs, files in os.walk(subdir):
                    if root == subdir:
                        if dirs:
                            assert all(d[:5] == 'shard' for d in dirs)
                            shards_begin += (max(int(d[5:]) for d in dirs),)
                        elif files:
                            assert all(f[:5] == 'shard' for f in files)
                            shards_begin += (max(int(f[5:f.index('.')]) for f in files),)
                        else:
                            shards_begin += (0,)
        if not args.unmanaged:
            with open(specification_path, 'r') as filehandle:
                loaded_specification = json.load(filehandle)
                assert loaded_specification == specification, str(loaded_specification) + '\n' + str(specification)

    else:
        shards_begin = (0,) * len(directories)
        if not args.unmanaged and os.path.isdir(directory):
            sys.stdout.write('Delete content of directory {directory}? '.format(directory=directory))
            sys.stdout.flush()
            if args.yes:
                sys.stdout.write('y\n')
            elif util.negative_response(sys.stdin.readline()[:-1]):
                exit(0)
            shutil.rmtree(directory)
            os.makedirs(directory)
        elif not os.path.isdir(directory):
            os.makedirs(directory)
        if not args.unmanaged:
            with open(specification_path, 'w') as filehandle:
                json.dump(specification, filehandle)
        if len(directories) > 1:
            for subdir in directories:
                if not os.path.isdir(subdir):
                    os.makedirs(subdir)

    for mode, directory, num_shards, shard_begin in zip(modes, directories, shards, shards_begin):
        sys.stdout.write('{time} generate {dtype} {name}{mode} data...\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=dataset.type, name=dataset.name, mode=(' ' + mode if mode else '')))
        if num_shards is not None:
            sys.stdout.write('         0%  0/{shards}  (time per shard: n/a)'.format(shards=num_shards))
            sys.stdout.flush()

        if args.clevr_format:  # validation --> val
            questions = list()

        for shard in range(1, (1 if num_shards is None else num_shards) + 1):
            before = datetime.now()
            if num_shards is None:
                path = directory
                num_shards = 1
            else:
                path = os.path.join(directory, 'shard{}'.format(shard_begin + shard))

            generated = dataset.generate(n=args.instances, mode=mode, include_model=(args.include_model or args.clevr_format), alternatives=True)

            if generated is None:
                assert False

            elif args.clevr_format:
                from shapeworld.world import World
                from shapeworld.datasets.clevr_util import parse_program
                assert args.type == 'agreement'
                worlds = generated['world']
                captions = generated['caption']
                captions_length = generated['caption_length']
                captions_model = generated.get('caption_model')
                agreements = generated['agreement']
                for n in range(len(worlds)):
                    index = (shard - 1) * args.instances + n
                    filename = 'world_{}.png'.format(index)
                    image_bytes = BytesIO()
                    World.get_image(world_array=worlds[n]).save(image_bytes, format='png')
                    with open(os.path.join(directory, filename), 'wb') as filehandle:
                        filehandle.write(image_bytes.getvalue())
                    image_bytes.close()
                    id2word = dataset.vocabulary(value_type='language')
                    if 'alternatives' in generated:
                        captions_iter = zip(captions[n], captions_length[n], captions_model[n], agreements[n])
                    else:
                        captions_iter = zip((captions[n],), (captions_length[n],), (captions_model[n],), (agreements[n],))
                    for caption, caption_length, caption_model, agreement in captions_iter:
                        if agreement == 1.0:
                            answer = 'true'
                        elif agreement == 0.0:
                            answer = 'false'
                        else:
                            assert False
                            answer = 'maybe'
                        if caption_model is None:
                            program = None
                        else:
                            program = parse_program(model=caption_model)
                        questions.append(dict(
                            image_index=index,
                            program=program,
                            question_index=0,
                            image_filename=filename,
                            question_family_index=0,
                            split=mode,
                            answer=answer,
                            question=' '.join(id2word[caption[i]] for i in range(caption_length))
                        ))

            else:
                if args.features:
                    for value_name, value_type in dataset.values.items():
                        if value_type == 'world':
                            features = pretrained_model.features(images=generated[value_name])
                            generated[value_name + '_features'] = features

                dataset.serialize(path=path, generated=generated, archive=args.archive, html=args.html, image_format=('png' if args.png_format else 'bmp'), concat_worlds=args.concatenate_images)
                if args.tf_records:
                    tf_util.write_records(dataset=dataset, records=generated, path=path)

            after = datetime.now()
            sys.stdout.write('\r         {completed:.0f}%  {shard}/{shards}  (time per shard: {duration})'.format(completed=(shard * 100 / num_shards), shard=shard, shards=num_shards, duration=str(after - before).split('.')[0]))
            sys.stdout.flush()

        if args.clevr_format:
            with open(os.path.join(directory, 'captions.json'), 'w') as filehandle:
                json.dump({'questions': questions}, filehandle)

        sys.stdout.write('\n')
        sys.stdout.flush()

    if args.features:
        pretrained_model.close()

    sys.stdout.write('{time} data generation completed\n'.format(time=datetime.now().strftime('%H:%M:%S')))
    sys.stdout.flush()
