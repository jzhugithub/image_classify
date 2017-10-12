from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'output_path', '/tmp/tfmodel/', 'Path where the results are saved to.')

tf.app.flags.DEFINE_string(
    'test_path', None,
    'The directory where the test images are stored or an absolute path to a '
    'test image.')

tf.app.flags.DEFINE_integer(
    'test_number', None,
    'The number of images to test')

tf.app.flags.DEFINE_integer(
    'num_classes', 80, 'Number of classes.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'test_image_size', None, 'Test image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.test_path:
        raise ValueError('You must supply the test image directory or an absolute '
                         'path with --test_path')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        ##############################
        # Define placeholder to feed #
        ##############################
        image_string_tensor = tf.placeholder(tf.string)
        image_tensor = tf.image.decode_jpeg(image_string_tensor, channels=3)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        test_image_size = FLAGS.test_image_size or network_fn.default_image_size
        processed_image = image_preprocessing_fn(image_tensor, test_image_size, test_image_size)
        processed_images = tf.expand_dims(processed_image, 0)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(processed_images)
        values, indices = tf.nn.top_k(logits, 3)

        with tf.Session() as sess:
            ###################
            # Load checkpoint #
            ###################
            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)

            #############
            # Run model #
            #############
            if os.path.isdir(FLAGS.test_path):
                image_dir = FLAGS.test_path
                image_name_list = os.listdir(image_dir)
                if FLAGS.test_number is not None:
                    image_name_list = image_name_list[:FLAGS.test_number]
            else:
                image_dir = os.path.dirname(FLAGS.test_path)
                image_name_list = [FLAGS.test_path.split('/')[-1]]

            result = []
            count = 0
            for image_name in image_name_list:
                count += 1
                print('image %s' % count)

                image_path = os.path.join(image_dir, image_name)
                image_string = open(image_path, 'rb').read()

                predictions = sess.run(indices, feed_dict={image_string_tensor: image_string})
                predictions = np.squeeze(predictions)

                result.append({'image_id': image_name, 'label_id': predictions.tolist()})
                print('image %s is %d,%d,%d' % (image_name, predictions[0], predictions[1], predictions[2]))

            with open(FLAGS.output_path, 'w') as f:
                json.dump(result, f)
                print('write result json to %s\n'
                      'num is %d' % (FLAGS.output_path, len(result)))


if __name__ == '__main__':
    tf.app.run()
