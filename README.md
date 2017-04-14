ShapeWorld
==========

**Getting started:**

```
git clone --recursive https://github.com/AlexKuhnle/ShapeWorld.git
pip3 install -e ShapeWorld
```



- [About ShapeWorld](#about-shapeworld)
- [Integration into Python code](#integration-into-python-code)
- [Stand-alone data generation](#stand-alone-data-generation)
- [Loading extracted data](#loading-extracted-data)
- [Example models](#example-models)



About ShapeWorld
----------------

ShapeWorld is a framework for specifying visual or multimodal datasets -- more precisely, *data generators*. It is based on the concept of *microworlds*, i.e. small and self-contained artificial scenarios consisting of colored shapes.

The main motivation behind ShapeWorld is to provide a new testbed and methodology for multimodal deep learning models. It differs from standard evaluation datasets in two ways: Firstly, data is randomly sampled during training/evaluation according to constraints specified by the experimenter. Secondly, its focus of evaluation is on linguistic understanding capabilities of the type investigated by formal semantics. In this context, the ShapeWorld tasks can be thought of as unit-testing multimodal models for specific linguistic generalization abilities -- similar to, for instance, the [bAbI tasks](https://research.fb.com/projects/babi/) of [Weston et al. (2015)](https://arxiv.org/abs/1502.05698) for text-only understanding.

![ShapeWorld](https://www.cl.cam.ac.uk/~aok25/files/shapeworld.png)

The code is written in Python 3. The data can either be obtained within a Python 3 module as [NumPy](http://www.numpy.org/) arrays, and hence integrates into deep learning projects based on common frameworks like [TensorFlow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/), or it can be extracted into separate files. Both options are described further below. For language generation, the Python package [pydmrs](https://github.com/delph-in/pydmrs) [(Copestake et al., 2016)](http://www.lrec-conf.org/proceedings/lrec2016/pdf/634_Paper.pdf) is required.

**The ShapeWorld framework is still under active development.**

I am interested in hearing about any applications you plan to use the ShapeWorld data for. In particular, let me know if you have a great idea in mind that you are interested in investigating with such abstract data, but which the current setup does not allow to realise -- I am happy to collaboratively find a way to make it happen.

Contact: aok25 (at) cam.ac.uk



Integration into Python code
----------------------------

The easiest way to use the ShapeWorld data in your Python 3 project is to directly call it from the code. Whenever a batch of training/evaluation instances is required, `dataset.generate(...)` is called with the respective arguments. This means that generation happens simultaneously to training/testing. Below an example of how to generate a batch of 128 training instances. See also the example models below.

```python
from shapeworld import Dataset

dataset = Dataset.from_config(
    dataset_type='agreement',
    dataset_name='multishape')
generated = dataset.generate(n=128, mode='train', include_models=True)

# given to the image caption agreement system
batch = (generated['world'], generated['caption'], generated['agreement'])

# can be used for more specific evaluation
world_model = generated['world-model']
caption_model = generated['caption-model']
```



Stand-alone data generation
---------------------------

The `shapeworld/generate.py` module provides options to generate ShapeWorld data in separate files via the command line. Use cases include:

* for using the ShapeWorld data for deep learning projects based on other programming languages than Python 3, like [Torch](http://torch.ch/),
* for quicker pre-evaluation of models in development stage -- however, it is one of the important principles of the ShapeWorld evaluation methodology and hence recommended to use newly generated data for the final evaluation phase,
* for manual investigation of the generated data,
* for entirely different applications like, e.g., a user study.

The following command line arguments are available:

* `--[d]irectory`:  Directory for generated data with automatically created sub-directories, should be non-existing or empty since it will be overwritten
* `--directory-[u]unmanaged`:  Directory for generated data, without automatically created sub-directories (default: `none`)
* `--[a]rchive`:  Store generated data in archive, either `zip[:mode]` or `tar[:mode]` with one of `none`, `deflate` (only zip), `gzip` (only tar), `bzip2`, `lzma` (default: `none`)
* `--[A]ppend`:  Append to existing data, when used with --directory (default: `false`)
* `--[t]ype`:  Dataset type (default: `agreement`)
* `--[n]ame`:  Dataset name (default: `oneshape`)
* `--[c]onfig`:  Dataset configuration file, otherwise use default configuration (default: `none`)
* `--[p]arts`:  Number of files (instead of all in one file), either a number or a triple of numbers, like `100,10,10` (requires mode `none`), for `train`, `validation` and `test` respectively (default: `1`)
* `--[i]nstances`:  Number of instances, per generated file (default: `100`)
* `--[m]ode`:  Mode, one of `train`, `validation`, `test` (default: `none`)
* `--include-[M]odels`:  Include world/caption models, as json file (default: `false`)
* `--no-[P]ixel-noise`:  Do not infuse pixel noise (default: `false`)
* `--captioner-[S]tatistics`:  Collect statistical data of captioner (default: `false`)

When creating larger amounts of ShapeWorld data, it is advisable to store the data in a compressed archive (for example `-a tar:bz2`) and turn off the pixel noise (`-p`) for best compression results. For instance, the following command line generates one million *training* instances of the `multishape` configuration file included in this repository:

```bash
python3 generate.py -D [DIRECTORY] -a tar:bzip2 -c configs/agreement/multishape.json -m train -p 100 -i 10k -W -P
```

For the purpose of this introduction, we generate a smaller amount of *all* training, validation and test instances using the default configuration of the dataset:

```bash
python3 generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -p 5,1,1 -i 128 -P
```



Loading extracted data
----------------------

Extracted data can be loaded and accessed with the same `Dataset` interface as before, just define the `dataset_name` as `'load'` and either the directory or the specification file as `config`. However, to be able to do this, we need to extract all of training, validation and test data, as is done in the last command line. Note that we extracted pixel-noise-free instances - the noise will automatically be (re-)infused accordingly.

```python
from shapeworld import Dataset

dataset = Dataset.from_config(
    dataset_name='load',
    config='examples/readme/agreement/multishape')
generated = dataset.generate(n=128, mode='train')
```

If you need to manually (re-)infuse the pixel noise later (for instance, because you want to load the data from another programming language), a procedure equivalent to the one used in the ShapeWorld framework can be used, which in Python 3 code looks the following:

```python
import numpy as np
from shapeworld import Dataset

dataset = Dataset.from_config(
    dataset_name='load',
    config='examples/readme/agreement/multishape')
world_size = 64
noise_range = 0.1
generated = dataset.generate(n=128, mode='train', noise=False)
worlds = generated['world']
for world in worlds:
    noise = np.random.normal(
        loc=0.0,
        scale=noise_range,
        size=(world_size, world_size, 3))
    mask = (noise < -noise_range) + (noise > noise_range)
    while np.any(a=mask):
        noise -= mask * noise
        noise += mask * np.random.normal(
            loc=0.0,
            scale=noise_range,
            size=(world_size, world_size, 3))
        mask = (noise < -noise_range) + (noise > noise_range)
    world += noise
    np.clip(world, a_min=0.0, a_max=1.0, out=world)
```



Example models
---------------

The `models/` directory contains a few exemplary models, which can be either applied directly or used as basis to build more sophisticated models. For direct application, the following command line arguments are available:

* `--[t]ype`:  Dataset type (default: `agreement`)
* `--[n]ame`:  Dataset name (default: `oneshape`)
* `--[c]onfig`:  Dataset configuration file, otherwise use default configuration (default: `none`)
* `--[m]odel`:  Model, one in `models/[TYPE]/` (default: `cnn_lstm_mult`)
* `--[l]earning-rate`:  Learning rate (default: `0.001`)
* `--[d]ropout-rate`:  Dropout rate (default: `0.0`)
* `--[p]arameters`:  Model parameters (default: `none`)
* `--[b]atch-size`:  Batch size (default: `128`)
* `--[i]terations`:  Number of training iterations (default: `1000`)
* `--[e]valuation-size`:  Evaluation size (default: `1024`)
* `--e[v]aluation-frequency`:  Evaluation frequency (default: `100`)
* `--model-[f]ile`:  Model file (default: `none`)
* `--[r]eport-file`:  CSV file reporting the evaluation results throughout the learning process (default: `none`)
* `--[R]estore`:  Restore system, requires `--model-file` (default: `false`)
* `--[E]valuate`:  Evaluate system without training, requires `--model-file` (default: `false`)

For instance, the following command line trains an image caption agreement system on the dataset specified by the `multishape` configuration file included in this repository:

```bash
python3 evaluate.py -t agreement -n multishape -m cnn_bow_mult -i 5k
```

The previously generated data can be loaded in the same way as was explained for loading the data in Python code:

```bash
python3 evaluate.py -t agreement -n load -c examples/readme/agreement/multishape -m cnn_bow_mult -i 10
```
