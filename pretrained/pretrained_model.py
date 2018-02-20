import os
import sys
import numpy as np
import tensorflow as tf

# current sub-directory
directory = os.path.dirname(os.path.realpath(__file__))

# add tf_models.research.slim submodule to Python path
sys.path.insert(1, os.path.join(directory, 'tf_models', 'research', 'slim'))

from pretrained.tf_models.research.slim.nets import nets_factory


# inception_v1_2016_08_28
# inception_v2_2016_08_28
# inception_v3_2016_08_28
# inception_v4_2016_09_09
# inception_resnet_v2_2016_08_30
# resnet_v1_50_2016_08_28
# resnet_v1_101_2016_08_28
# resnet_v1_152_2016_08_28
# resnet_v2_50_2017_04_14
# resnet_v2_101_2017_04_14
# resnet_v2_152_2017_04_14
# TBA
# vgg_16_2016_08_28
# vgg_19_2016_08_28
# mobilenet_v1_1.0_224_2017_06_14
# mobilenet_v1_0.50_160_2017_06_14
# mobilenet_v1_0.25_128_2017_06_14
# nasnet-a_mobile_04_10_2017
# nasnet-a_large_04_10_2017

class PretrainedModel(object):

    def __init__(self, image_shape, batch_size=64, model='resnet_v2_101_2017_04_14', tensor='resnet_v2_101/block3/unit_22/bottleneck_v2/add:0'):
        self.batch_size = batch_size

        self.images = tf.placeholder(dtype=tf.float32, shape=(None,) + image_shape)
        name = model.rsplit('_', 3)[0]
        network_fn = nets_factory.get_network_fn(name=name, num_classes=1001)
        size = (network_fn.default_image_size, network_fn.default_image_size)
        images = tf.image.resize_images(images=self.images, size=size, method=tf.image.ResizeMethod.BICUBIC, align_corners=False)
        network_fn(images=images)  # logits, end_points =

        downloads_path = os.path.join(directory, 'downloads')
        model_path = os.path.join(downloads_path, name + '.ckpt')

        if not os.path.isdir(downloads_path):
            os.mkdir(downloads_path)

        if not os.path.isfile(model_path):
            import tarfile
            import wget
            tar_path = os.path.join(downloads_path, model + '.tar.gz')
            wget.download(url='http://download.tensorflow.org/models/{}.tar.gz'.format(model), out=tar_path)
            tar = tarfile.open(tar_path, "r:gz")
            tar.extractall(path=downloads_path)
            tar.close()
            os.remove(tar_path)

        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path=model_path, var_list=tf.contrib.framework.get_variables())

        self.session = tf.Session()
        init_fn(self.session)
        # tf.summary.FileWriter(logdir=download_path, graph=session.graph)

        self.tensor = self.session.graph.get_tensor_by_name(name=tensor)
        self.features_shape = tuple(self.tensor.get_shape().as_list()[1:])

    def features(self, images):
        features_list = list()
        for n in range(len(images) // self.batch_size):
            features_list.append(self.session.run(fetches=self.tensor, feed_dict={self.images: images[n * self.batch_size: (n + 1) * self.batch_size]}))
        if len(images) % self.batch_size > 0:
            features_list.append(self.session.run(fetches=self.tensor, feed_dict={self.images: images[-(len(images) % self.batch_size):]}))
        features = np.concatenate(features_list, axis=0)
        assert features.shape[0] == images.shape[0]
        return features

    def close(self):
        self.session.close()
