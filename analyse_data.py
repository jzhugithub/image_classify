from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import skimage.io
import os
import matplotlib.pyplot as plt
import numpy as np


def get_label_image_dict(annotation_json_path):
    '''
    load json, return dictionary of label-image_name
    :param annotation_json_path: json path of annotation
    :return: dictionary of label-image_name
    '''
    with open(annotation_json_path, 'r') as f:
        items = json.load(f)
    label_image_dict = {}
    for item in items:
        label = int(item['label_id'])
        image = item['image_id']
        if label not in label_image_dict.keys():
            label_image_dict[label] = []
        label_image_dict[label].append(image)
    return label_image_dict


def show_image(annotation_json_path, image_dir, label):
    assert label >= 0 and label <= 79
    label_image_dict = get_label_image_dict(annotation_json_path)
    for image_name in label_image_dict[label]:
        image_path = os.path.join(image_dir, image_name)
        print('label: %s' % label)
        print('image_path: %s' % image_path)
        image = skimage.io.imread(image_path)
        skimage.io.imshow(image)
        skimage.io.show()

def count_image_number(annotation_json_path):
    label_image_dict = get_label_image_dict(annotation_json_path)

    labels = np.zeros(len(label_image_dict))
    counts = np.zeros(len(label_image_dict))
    i = 0
    for label in label_image_dict:
        labels[i] = label
        counts[i] = len(label_image_dict[label])
        i += 1

    plt.bar(labels, counts)
    plt.show()


if __name__ == '__main__':
    annotation_json_path = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json'
    image_dir = '/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_train_20170904/scene_train_images_20170904'

    count_image_number(annotation_json_path)
    show_image(annotation_json_path, image_dir, label=79)


