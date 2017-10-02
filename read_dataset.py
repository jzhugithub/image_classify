from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

slim = tf.contrib.slim

# modify
_FILE_PATTERN = 'data_%s_*.tfrecord'
_LABEL_TXT_NAME = 'labels.txt'
SPLITS_TO_SIZES = {'train': 53879, 'validation': 7120}
NUM_CLASSES = 80
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 79',
}


def get_dataset(split_name, dataset_dir):
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern = os.path.join(dataset_dir, _FILE_PATTERN % split_name)

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None
    labels_txt_path = os.path.join(dataset_dir, _LABEL_TXT_NAME)
    if os.path.exists(labels_txt_path):
        labels_to_names = {}
        with open(labels_txt_path, 'rb') as f:
            for line in f:
                label, name = line.strip().split(':')
                labels_to_names[int(label)] = name

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=labels_to_names)
