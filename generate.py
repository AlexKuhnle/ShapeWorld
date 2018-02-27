import argparse
from datetime import datetime
from io import BytesIO
import json
import os
import shutil
import sys
from shapeworld import Dataset, util
from shapeworld.world import World


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate example data')

    parser.add_argument('-d', '--directory', help='Directory for generated data (with automatically created sub-directories, unless --unmanaged)')
    parser.add_argument('-a', '--archive', default=None, choices=('zip', 'zip:none', 'zip:deflate', 'zip:bzip2', 'zip:lzma', 'tar', 'tar:none', 'tar:gzip', 'tar:bzip2', 'tar:lzma'), help='Store generated data in (compressed) archives')
    parser.add_argument('-A', '--append', action='store_true', help='Append to existing data (when used without --unmanaged)')
    parser.add_argument('-U', '--unmanaged', action='store_true', help='Do not automatically create sub-directories (implied if --files not specified)')

    parser.add_argument('-t', '--type', default=None, help='Dataset type')
    parser.add_argument('-n', '--name', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Dataset name')
    parser.add_argument('-l', '--language', default=None, help='Dataset language')
    parser.add_argument('-c', '--config', type=util.parse_tuple(parse_item=str, unary_tuple=False), default=None, help='Dataset configuration file')

    parser.add_argument('-f', '--files', type=util.parse_tuple(parse_item=util.parse_int_with_factor, unary_tuple=True, valid_sizes=(1, 3)), default=None, help='Number of files to split data into (not specified implies --unmanaged)')
    parser.add_argument('-s', '--start', type=util.parse_tuple(parse_item=util.parse_int_with_factor, unary_tuple=True, valid_sizes=(1, 3)), default=None, help='Start file number (requires --append)')
    parser.add_argument('-m', '--mode', default=None, choices=('train', 'validation', 'test'), help='Mode')
    parser.add_argument('-i', '--instances', type=util.parse_int_with_factor, default=128, help='Number of instances per file')

    parser.add_argument('-P', '--delay-pixel-noise', action='store_true', help='Do not infuse pixel noise now, but when dataset is loaded')
    parser.add_argument('-M', '--include-model', action='store_true', help='Include world/caption model (as json file)')
    parser.add_argument('-H', '--html', action='store_true', help='Create HTML file visualizing the generated data')
    parser.add_argument('-T', '--tf-records', action='store_true', help='Additionally store data as TensorFlow records')
    parser.add_argument('-F', '--features', action='store_true', help='Additionally extract image features (conv4 of resnet_v2_101)')
    parser.add_argument('-C', '--clevr-format', action='store_true', help='Output in CLEVR format')
    parser.add_argument('-O', '--concatenate-images', action='store_true', help='Concatenate images per part into one image file')

    parser.add_argument('-Y', '--yes', action='store_true', help='Confirm all questions with yes')
    # parser.add_argument('-v', '--values', default=None, help='Comma-separated list of values to include')

    parser.add_argument('--config-values', nargs=argparse.REMAINDER, default=(), help='Additional dataset configuration values passed as command line arguments')
    args = parser.parse_args()
    args.config_values = util.parse_config(values=args.config_values)

    if args.tf_records:
        from shapeworld import tf_util

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
        sys.stdout.write('{time} warning: part size is {size}MB '.format(time=datetime.now().strftime('%H:%M:%S'), size=int(args.instances * util.product(dataset.world_shape()) / 1e6)))
        sys.stdout.flush()
        if args.yes:
            sys.stdout.write('y\n')
        elif util.negative_response(sys.stdin.readline()[:-1]):
            exit(0)

    if args.features:
        assert args.tf_records
        from pretrained import PretrainedModel
        pretrained_model = PretrainedModel(image_shape=dataset.world_shape())
        for value_name, value_type in list(dataset.values.items()):
            if value_type == 'world':
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
    if args.concatenate_images:
        specification['num_concat_worlds'] = args.instances

    args.unmanaged = args.unmanaged or (args.files is None)
    if args.unmanaged:
        directory = args.directory
        if args.files is None:
            parts = (0,)
        else:
            parts = args.files

    else:
        if dataset.language is None:
            directory = os.path.join(args.directory, dataset.type, dataset.name)
            specification_path = os.path.join(args.directory, '{}-{}.json'.format(dataset.type, dataset.name))
        else:
            directory = os.path.join(args.directory, '{}-{}'.format(dataset.type, dataset.language), dataset.name)
            specification_path = os.path.join(args.directory, '{}-{}-{}.json'.format(dataset.type, dataset.language, dataset.name))
        parts = args.files

    assert all(part >= 0 for part in parts)
    assert args.files is None or (args.mode is not None) == (len(parts) == 1)
    if len(parts) == 1:
        modes = (args.mode,)
        if args.unmanaged:
            directories = (directory,)
        else:
            directories = (os.path.join(directory, args.mode),)
    elif len(parts) == 3:
        assert args.mode is None
        modes = ('train', 'validation', 'test')
        directories = tuple(os.path.join(directory, mode) for mode in modes)

    assert args.start is None or args.append
    assert args.start is None or len(args.start) == len(args.files)
    if args.append:
        if args.start is not None:
            start_part = args.start
        elif args.clevr_format:
            start_part = tuple(0 for _ in directories)
        else:
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
                loaded_specification = json.load(filehandle)
                assert loaded_specification == specification, str(loaded_specification) + '\n' + str(specification)

    else:
        start_part = (0,) * len(directories)
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

    for mode, directory, num_parts, start in zip(modes, directories, parts, start_part):
        sys.stdout.write('{time} generate {dtype} {name}{mode} data...\n'.format(time=datetime.now().strftime('%H:%M:%S'), dtype=dataset.type, name=dataset.name, mode=(' ' + mode if mode else '')))
        sys.stdout.write('         0%  0/{files}  (time per part: n/a)'.format(files=num_parts))
        sys.stdout.flush()

        if args.clevr_format:  # validation --> val
            questions = list()

        for part in range(1, num_parts + (num_parts == 0) + 1):
            before = datetime.now()
            if num_parts == 0:
                path = directory
                num_parts = 1
            else:
                path = os.path.join(directory, 'part{}'.format(start + part))

            generated = dataset.generate(n=args.instances, mode=mode, include_model=(args.include_model or args.clevr_format), alternatives=True)

            if generated is None:
                assert False

            elif args.clevr_format:

                def parse(model, index, inputs):
                    if model['component'] == 'Attribute':
                        return [dict(inputs=inputs, function='attribute', value_inputs=['{}-{}'.format(model['predtype'], model['value'])])]
                    elif model['component'] == 'EntityType':
                        outputs = [dict(inputs=list(), function='scene', value_inputs=list())]
                        for model in model['value']:
                            outputs += parse(model=model, index=(index + len(outputs)), inputs=[index + len(outputs) - 1])
                        return outputs
                    elif model['component'] == 'Relation':  # should connect relation???
                        if model['predtype'] in ('attribute', 'type'):
                            value = parse(model=model['value'], index=index, inputs=list())
                            return value + [dict(inputs=[index + len(value) - 1], function='relation', value_inputs=[model['predtype']])]
                        else:
                            reference = parse(model=model['reference'], index=index, inputs=list())
                            if 'comparison' in model:
                                comparison = parse(model=model['comparison'], index=(index + len(reference)), inputs=list())
                                return reference + comparison + [dict(inputs=[index + len(reference) - 1, index + len(reference) + len(comparison) - 1], function='relation', value_inputs=['{}({})'.format(model['predtype'], model['value'])])]
                            else:
                                return reference + [dict(inputs=[index + len(reference) - 1], function='relation', value_inputs=['{}-{}'.format(model['predtype'], model['value'])])]
                    elif model['component'] == 'Existential':  # should connect relation???
                        restrictor = parse(model=model['restrictor'], index=index, inputs=list())
                        body = parse(model=model['body'], index=(index + len(restrictor)), inputs=[index + len(restrictor) - 1])
                        return restrictor + body + [dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(body) - 1], function='quantifier', value_inputs=['existential'])]
                    elif model['component'] == 'Quantifier':
                        restrictor = parse(model=model['restrictor'], index=index, inputs=list())
                        body = parse(model=model['body'], index=(index + len(restrictor)), inputs=[index + len(restrictor) - 1])
                        return restrictor + body + [dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(body) - 1], function='quantifier', value_inputs=['{}-{}-{}'.format(model['qtype'], model['qrange'], model['quantity'])])]
                    elif model['component'] == 'NumberBound':
                        quantifier = parse(model=model['quantifier'], index=index, inputs=list())
                        return quantifier + [dict(inputs=[index + len(quantifier) - 1], function='number-bound', value_inputs=[str(model['bound'])])]
                    elif model['component'] == 'ComparativeQuantifier':
                        restrictor = parse(model=model['restrictor'], index=index, inputs=list())
                        comparison = parse(model=model['comparison'], index=(index + len(restrictor)), inputs=list())
                        restrictor_body = parse(model=model['body'], index=(index + len(restrictor) + len(comparison)), inputs=[index + len(restrictor) - 1])
                        comparison_body = parse(model=model['body'], index=(index + len(restrictor) + len(comparison)), inputs=[index + len(restrictor) + len(comparison) - 1])
                        return restrictor + comparison + restrictor_body + comparison_body + [dict(inputs=[index + len(restrictor) - 1, index + len(restrictor) + len(comparison) - 1, index + len(restrictor) + len(comparison) + len(restrictor_body) - 1, index + len(restrictor) + len(comparison) + len(restrictor_body) + len(comparison_body) - 1], function='comparative-quantifier', value_inputs=['{}-{}-{}'.format(model['qtype'], model['qrange'], model['quantity'])])]
                    elif model['component'] == 'Proposition':
                        assert False, model
                    else:
                        assert False, model

                assert args.type == 'agreement'
                worlds = generated['world']
                captions = generated['caption']
                captions_length = generated['caption_length']
                captions_model = generated.get('caption_model')
                agreements = generated['agreement']
                for n in range(len(worlds)):
                    index = (part - 1) * args.instances + n
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
                        questions.append(dict(
                            image_index=index,
                            program=(None if caption_model is None else parse(model=caption_model, index=0, inputs=list())),
                            question_index=0,
                            image_filename=filename,
                            question_family_index=0,
                            split=mode,
                            answer=answer,
                            question=' '.join(id2word[caption[i]] for i in range(caption_length))
                        ))

            else:
                dataset.serialize(path=path, generated=generated, archive=args.archive, concat_worlds=args.concatenate_images, html=args.html)
                if args.tf_records:
                    if args.features:
                        for value_name, value_type in dataset.values.items():
                            if value_type == 'world':
                                features = pretrained_model.features(images=generated[value_name])
                                generated[value_name + '_features'] = features
                    tf_util.write_records(dataset=dataset, records=generated, path=path)

            after = datetime.now()
            sys.stdout.write('\r         {completed:.0f}%  {part}/{parts}  (time per part: {duration})'.format(completed=((part) * 100 / num_parts), part=part, parts=num_parts, duration=str(after - before).split('.')[0]))
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
