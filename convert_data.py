from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from datasets import dataset_utils
import math
import os
import tensorflow as tf
import json
import csv


def get_image_label_list(annotation_json_path):
    '''
    load json, return list with image name and label
    :param annotation_json_path: json path of annotation
    :return: list with image name and label
    '''
    with open(annotation_json_path, 'r') as f:
        items = json.load(f)
    image_label_list = []
    for item in items:
        image_label = [item['image_id'], item['label_id']]
        image_label_list.append(image_label)
    return image_label_list


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def convert_dataset(image_dir, annotation_json_path, output_dir, split_name, _NUM_SHARDS=5):
    print(type(annotation_json_path))
    image_label_list = get_image_label_list(annotation_json_path)

    num_per_shard = int(math.ceil(len(image_label_list) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_path = os.path.join(output_dir,
                                           'data_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS))
                with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(image_label_list))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image {}/{} shard {}'.format(
                            i + 1, len(image_label_list), shard_id))
                        sys.stdout.flush()
                        image_data = tf.gfile.FastGFile(os.path.join(image_dir, image_label_list[i][0]), 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, int(image_label_list[i][1]))
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def label_csv2txt(cvs_path, output_dir):
    with open(cvs_path, 'rb') as csv_in:
        with open(os.path.join(output_dir, 'labels.txt'), 'w') as txt_out:
            for row in csv.reader(csv_in):
                txt_out.write('%s:%s\n' % (row[0], row[2]))


if __name__ == '__main__':
    # modify
    label_csv_path = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_train_20170904/scene_classes.csv'
    image_dir_train = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_train_20170904/scene_train_images_20170904'
    annotation_json_path_train = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
    image_dir_val = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_validation_20170908/scene_validation_images_20170908'
    annotation_json_path_val = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json'
    output_dir = '/home/zj/database_temp/ai_challenger_scene/tfrecord'
    num_shards = 5

    os.system('mkdir -p %s' % output_dir)
    # labels
    label_csv2txt(label_csv_path, output_dir)
    # train
    convert_dataset(
        image_dir=image_dir_train,
        annotation_json_path=annotation_json_path_train,
        output_dir=output_dir,
        split_name='train',
        _NUM_SHARDS=num_shards)
    # validation
    convert_dataset(
        image_dir=image_dir_val,
        annotation_json_path=annotation_json_path_val,
        output_dir=output_dir,
        split_name='validation',
        _NUM_SHARDS=num_shards)
