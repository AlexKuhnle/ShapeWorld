# ShapeWorld

### Getting started

```bash
git clone --recursive https://github.com/AlexKuhnle/ShapeWorld.git
pip3 install -e ShapeWorld
```


### Recently added features

- More complex quantification captions
- Abstract caption model as sequence in reverse Polish notation
- Improved sampling strategy for more balanced / less biased data
- Generator option `-H` to create an HTML file displaying the generated data (e.g. see in (examples))
- Various new and extended caption agreement datasets


### Table of content

- [About ShapeWorld](#about-shapeworld)
- [Example data](#example-data)
- [Integration into Python code](#integration-into-python-code)
- [Stand-alone data generation](#stand-alone-data-generation)
- [Loading extracted data](#loading-extracted-data)
- [CLEVR and NLVR interface](#clevr-and-nlvr-interface)
- [Evaluation and example models](#evaluation-and-example-models)



## About ShapeWorld

ShapeWorld is a framework which allows to specify generators for abstract, visually grounded language data (or just visual data).

The main motivation behind ShapeWorld is to provide a new testbed and evaluation methodology for visually grounded language understanding, particularly aimed at deep learning models. It differs from standard evaluation datasets in two ways: Firstly, data is randomly sampled during training and evaluation according to constraints specified by the experimenter. Secondly, its focus of evaluation is on linguistic understanding capabilities of the type investigated by formal semantics. In this context, the ShapeWorld tasks can be thought of as unit-testing multimodal models for specific linguistic generalization abilities -- similar to, for instance, the [bAbI tasks](https://research.fb.com/projects/babi/) of [Weston et al. (2015)](https://arxiv.org/abs/1502.05698) for text-only understanding.

The code is written in Python 3 (but should be compatible to Python 2). The data can either be obtained within a Python module as [NumPy](http://www.numpy.org/) arrays, and hence integrates into deep learning projects based on common frameworks like [TensorFlow](https://www.tensorflow.org/), [PyTorch](http://pytorch.org/) or [Theano](http://deeplearning.net/software/theano/), or it can be extracted into separate files. Both options are described further below. For language generation, the Python package [pydmrs](https://github.com/delph-in/pydmrs) [(Copestake et al., 2016)](http://www.lrec-conf.org/proceedings/lrec2016/pdf/634_Paper.pdf) is required.

**The ShapeWorld framework is still under active development.**

I am interested in hearing about any applications you plan to use the ShapeWorld data for. In particular, let me know if you have a great idea in mind that you are interested in investigating with such abstract data, but which the current setup does not allow to realize -- I am happy to collaboratively find a way to make it happen.

Contact: aok25 (at) cam.ac.uk

If you use ShapeWorld in your work, please cite:

> **ShapeWorld: A new test methodology for multimodal language understanding** ([arXiv](https://arxiv.org/abs/1704.04517))
> *Alexander Kuhnle and Ann Copestake* (April 2017)



## Example data

#### Caption agreement datasets

- [Oneshape (simple)](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/oneshape_simple/data.html)
- [Oneshape](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/oneshape/data.html)
- [Multishape (simple)](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/multishape_simple/data.html)
- [Multishape](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/multishape/data.html)
- [Spatial (simple)](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/spatial_simple/data.html)
- [Spatial](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/spatial/data.html)
- [Relational](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/relational/data.html)
- [MaxAttributes](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/maxattr/data.html)
- [Count Quantification (simple)](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/quantification_count_simple/data.html)
- [Count Quantification](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/quantification_count/data.html)
- [Ratio Quantification (simple)](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/quantification_ratio_simple/data.html)
- [Ratio Quantification](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/quantification_ratio/data.html)
- [Quantification (simple)](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/quantification_simple/data.html)
- [Combination](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/agreement/combination/data.html)

#### Classification datasets

- [Oneshape](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/classification/oneshape/data.html)
- [Multishape](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/classification/multishape/data.html)
- [Countshape](https://rawgit.com/AlexKuhnle/ShapeWorld/master/examples/classification/countshape/data.html)



## Integration into Python code

The easiest way to use the ShapeWorld data in your Python project is to directly call it from the code. Whenever a batch of training/evaluation instances is required, `dataset.generate(...)` is called with the respective arguments. This means that generation happens simultaneously to training/testing. Below an example of how to generate a batch of 100 training instances. See also the example models below.

```python
from shapeworld import dataset

dataset = dataset(dtype='agreement', name='multishape')
generated = dataset.generate(n=100, mode='train', noise_range=0.1, include_model=True)

# given to the image caption agreement system
batch = (generated['world'], generated['caption'], generated['agreement'])

# can be used for more specific evaluation
world_model = generated['world_model']
caption_model = generated['caption_model']
```



## Stand-alone data generation

The `shapeworld/generate.py` module provides options to generate ShapeWorld data in separate files via the command line. Use cases include:

* for using the ShapeWorld data for deep learning projects based on other programming languages than Python,
* for quicker pre-evaluation of models in development stage -- however, it is one of the important principles of the ShapeWorld evaluation methodology and hence recommended to use newly generated data for the final evaluation phase,
* for manual investigation of the generated data,
* for entirely different applications like, e.g., a user study.

The following command line arguments are available:

* `--[d]irectory`:  Directory for generated data, with automatically created sub-directories unless `--unmanaged`, hence should be non-existing or empty since it will be overwritten (**required**)
* `--[a]rchive`:  Store generated data in (compressed) archives, either `zip[:mode]` or `tar[:mode]` with one of `none`, `deflate` (only zip), `gzip` (only tar), `bzip2`, `lzma` (default: `none`)
* `--[A]ppend`:  Append to existing data (when used without `--unmanaged`)
* `--[U]nmanaged`:  Do not automatically create sub-directories (implied if --files not specified)

* `--[t]ype`:  Dataset type (default: `agreement`)
* `--[n]ame`:  Dataset name (**required**)
* `--[l]anguage`:  Dataset language, if available (default: `none`, i.e. English)
* `--[c]onfig`:  Dataset configuration file, otherwise use default configuration

* `--[m]ode`:  Mode, one of `train`, `validation`, `test` or `tf-records`, requires `--files` to be a single number (default: `none`)
* `--[f]iles`:  Number of files to split data into instead of all in one file (not specified implies --unmanaged), either a number (requires `--mode`), or a tuple of 3 (or 4) numbers like `(100,10,10)` (without `--mode`), for (`tf-records`,) `train`, `validation` and `test` data respectively (default: `1`)
* `--[i]nstances`:  Number of instances per file (default: `100`)

* `--[p]ixel-noise`: Pixel noise range (default: `0.0`)
* `--include-[M]odel`:  Include world/caption model (as json file)
* `--[C]oncatenate-images`:  Concatenate images per part into one image file
* `--[H]tml`:  Create HTML file (`data.html`) displaying the generated data

When creating larger amounts of ShapeWorld data, it is advisable to store the data in a compressed archive (for example `-a tar:bz2`) and turn off the pixel noise (`-p`) for best compression results. For instance, the following command line generates one million *training* instances of the `multishape` configuration file included in this repository:

```bash
python generate.py -D [DIRECTORY] -a tar:bzip2 -t agreement -n multishape -m train -f 100 -i 10k -M
```

For the purpose of this introduction, we generate a smaller amount of *all* training (TensorFlow records and raw), validation and test instances using the default configuration of the dataset:

```bash
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -f "(5,5,1,1)" -i 100 -M
```



## Loading extracted data

Extracted data can be loaded and accessed with the same `Dataset` interface as before, just define the `config` argument as `'load([DIRECTORY])'`. However, to be able to do this, we need to extract all of training, validation and test data, as is done in the last command line. Note that we extracted pixel-noise-free instances - the noise will automatically be (re-)infused accordingly.

```python
from shapeworld import dataset

dataset = dataset(dtype='agreement', name='multishape', config='load(examples/readme)')
generated = dataset.generate(n=100, mode='train', noise_range=0.1)
```

Loading the data in Python and then feeding it to a model is relatively slow. By using TensorFlow (TF) records (see above for how to generate TF records) and consequently the ability to load data implicitly within TensorFlow, models can be trained significantly faster. ShapeWorld provides utilities to access TF records as generated/loaded data would be handled:

```python
from shapeworld import tf_util

generated = tf_util.batch_records(dataset=dataset, batch_size=100, noise_range=0.1)
```

If you need to manually (re-)infuse the pixel noise later (for instance, because you want to load the data from another programming language), a procedure equivalent to the one used in the ShapeWorld framework can be used, which in Python code looks the following:

```python
import numpy as np
from shapeworld import dataset

dataset = dataset(dtype='agreement', name='multishape', config='load(examples/readme)')
world_size = 64
generated = dataset.generate(n=100, mode='train')
worlds = generated['world']
noise_range = 0.1
for world in worlds:
    noise = np.random.normal(
        loc=0.0,
        scale=noise_range,
        size=dataset.world_shape)
    mask = (noise < -2.0 * noise_range) + (noise > 2.0 * noise_range)
    while np.any(a=mask):
        noise -= mask * noise
        noise += mask * np.random.normal(
            loc=0.0,
            scale=noise_range,
            size=dataset.world_shape)
        mask = (noise < -2.0 * noise_range) + (noise > 2.0 * noise_range)
    world += noise
    np.clip(world, a_min=0.0, a_max=1.0, out=world)
```



## CLEVR and NLVR interface

CLEVR can be obtained as follows (alternatively, replace `CLEVR_v1.0` with `CLEVR_CoGenT_v1.0` for the CLEVR CoGenT dataset):

```bash
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
rm CLEVR_v1.0.zip
```

ShapeWorld then provides a basic interface to load the CLEVR instances **in order of their appearance** in the dataset. It is hence recommended to *'pre-generate'* the entire dataset (70k training, 15k validation and 15k test instances) once through the ShapeWorld interface, either as `clevr_classification` or `clevr_answering` dataset type, and subsequently access it as you would load other pre-generated ShapeWorld datasets:

```bash
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t clevr_classification -n clevr -c "{directory=CLEVR_v1.0}" -f "(140,0,30,30)" -i 500 -M
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t clevr_classification -n clevr -c "{directory=CLEVR_v1.0}" -m train -f 140 -i 500 -M -A
rm -r CLEVR_v1.0
```

Accordingly, in the case of CLEVR CoGenT:

```bash
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t clevr_classification -n clevr -c "{directory=CLEVR_CoGenT_v1.0,parts={train=A,validation=A,test=A}}" -f "(140,0,30,30)" -i 500 -M
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t clevr_classification -n clevr -c "{directory=CLEVR_CoGenT_v1.0,parts={train=A,validation=A,test=A}}" -m train -f 140 -i 500 -M -A
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t clevr_classification -n clevr -c "{directory=CLEVR_CoGenT_v1.0,parts={train=A,validation=B,test=B}}" -f "(0,30,30)" -i 500 -M
rm -r CLEVR_CoGenT_v1.0
```

As `clevr_classification` dataset, it provides:
- a `'world'` and optional `'world_model'` value,
- an integer `'alternatives'` value which gives the number of questions per image (usually 10),
- a `'question'`, `'question_length'` and optional `'question_model'` value array/list containing the respective number of actual question values,
- an `'answer'` integer array specifying the corresponding answers.

As `clevr_answering` dataset, the last value is replaced by:
- an `'answer'` and `'answer_length'` value array containing the corresponding answer values.

Equivalently, NLVR can be obtained via:

```bash
git clone https://github.com/cornell-lic/nlvr.git
```

Again, one should *'pre-generate'* the entire dataset (75k training, 6k validation and 6k test instances) as `nlvr_agreement` dataset type, and subsequently access it via the ShapeWorld load interface:

```bash
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t nlvr_agreement -n nlvr -c "{directory=nlvr}" -f "(25,0,2,2)" -i 3k -M
python generate.py -d [SHAPEWORLD_DIRECTORY] -a tar:bzip2 -t nlvr_agreement -n nlvr -c "{directory=nlvr}" -m train -f 25 -i 3k -M -A
rm -r nlvr
```

The dataset provides:
- `'world1'`, `'world2'`, `'world3'` and optional `'world_model1'`, `'world_model2'`, `'world_model3'` values,
- a `'description'`, `'description_length'` and optional `'description_model'` values,
- an `'agreement'` value.



## Evaluation and example models

The `models/` directory contains a few exemplary models based on [TFMacros](https://github.com/AlexKuhnle/TFMacros), my collection of TensorFlow macros. The scripts `train.py` and `evaluate.py` provide the following command line arguments to train and evaluate these models:

* `--[t]ype`:  Dataset type (**required**)
* `--[n]ame`:  Dataset name (**required**)
* `--[l]anguage`:  Dataset language, if available (default: `none`, i.e. English)
* `--[c]onfig`:  Dataset configuration file, otherwise use default configuration
* `--[p]ixel-noise`:  Pixel noise range (default: `0.1`)

* `--[m]odel`:  Model, one in `models/[TYPE]/` (**required**)
* `--h[y]perparams-file`:  Model hyperparameters file (default: hyperparams directory)
* `--learning-[r]ate`:  Learning rate (default: `0.0001`)
* `--[R]estore`:  Restore system, requires `--model-file` (default: `false`)

* `--[i]terations`:  Number of training iterations (default: `1000`)
* `--[b]atch-size`:  Batch size (default: `100`)
* `--[e]valuation-size`:  Evaluation size (default: `1000`)
* `--evaluation-[f]requency`:  Evaluation frequency (default: `100`)
* `--[q]uery`:  Additional values to query (separated by commas) (default: `none`)
* `--[T]f-records`:  Use TensorFlow records

* `--model-dir`:  TensorFlow model directory, storing the model computation graph and parameters
* `--summary-dir`:  TensorFlow summary directory for TensorBoard
* `--report-file`:  CSV file reporting the evaluation results throughout the learning process

* `--[v]erbosity`:  Verbosity, one of `0` (no messages), `1` (default), `2` (plus TensorFlow messages)

For instance, the following command line trains an image caption agreement system on the dataset specified by the `multishape` configuration file included in this repository:

```bash
python evaluate.py -t agreement -n multishape -m cnn_bow_mult -i 5k
```

The previously generated data (here: TF records) can be loaded in the same way as was explained for loading the data in Python code:

```bash
python evaluate.py -t agreement -n multishape -c "load(examples/readme)" -m cnn_bow -i 10 -T
```
