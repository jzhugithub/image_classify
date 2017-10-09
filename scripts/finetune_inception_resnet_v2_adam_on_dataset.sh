#!/bin/bash

################
# Before train #
################
# 1. convert dataset
# modify parameters in 'convert_data.py'
# cd to image_classify
# run: python convert_data.py

# 2. set dataset for model
# modify parameter in 'read_dataset.py'

###############
# Start train #
###############
# modify parameters in '{scripts}/script.sh'
# cd to image_classify/scripts
# run: bash {script}.sh

###################
# Run tensorboard #
###################
# tensorboard --logdir=${MODEL_DIR}
# http://localhost:6006

##############
# Result log #
##############
# --stage_2--
# train7000: 0.946
# val: 0.940
# test: ?

# --stage_3--
# train7000: 0.954
# val: 0.944
# test: 0.942

# Where the train and evaluation dataset is saved to.
TRAIN_EVAL_TFRECORD_DIR=/home/zj/database_temp/ai_challenger_scene/tfrecord

# Where the pre-trained inception_resnet_v2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_PATH=/home/zj/database_temp/inception_resnet_v2_2016_08_30/inception_resnet_v2_2016_08_30.ckpt

# Where the checkpoint and logs will be saved to.
MODEL_DIR=/home/zj/my_workspace/image_classify/models/inception_resnet_v2_adam

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${MODEL_DIR}/train

# Where the evaluation logs will be saved to.
EVAL_DIR=${MODEL_DIR}/eval

# Where the test dataset(a dictionary or an image path) is saved to.
TEST_DATASET_PATH=/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922

# Where the train dataset is saved to.
TRAIN_DATASET_PATH=/home/zj/database_temp/ai_challenger_scene/ai_challenger_scene_train_20170904/scene_train_images_20170904

# Where the test output file will be saved to.
TEST_DIR=${MODEL_DIR}/test
TEST_OUTPUT=submit_10_08_test_3.json
TEST_TRAIN_NUM=7000
TEST_TRAIN_OUTPUT=submit_10_08_train7000_3.json

# Model's class number.
NUM_CLASSES=80


# make dictionary to use.
mkdir -p ${TRAIN_DIR}/stage_1
mkdir -p ${EVAL_DIR}/stage_1
mkdir -p ${TRAIN_DIR}/stage_2
mkdir -p ${EVAL_DIR}/stage_2
mkdir -p ${TRAIN_DIR}/stage_3
mkdir -p ${EVAL_DIR}/stage_3
mkdir -p ${TEST_DIR}


#### stage_1 ####
# Fine-tune only the new layers for 3000 steps.
python ../train_classifier.py \
  --train_dir=${TRAIN_DIR}/stage_1 \
  --dataset_split_name=train \
  --dataset_dir=${TRAIN_EVAL_TFRECORD_DIR} \
  --model_name=inception_resnet_v2 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_PATH} \
  --checkpoint_exclude_scopes='InceptionResnetV2/AuxLogits, InceptionResnetV2/Logits' \
  --trainable_scopes='InceptionResnetV2/AuxLogits, InceptionResnetV2/Logits' \
  --max_number_of_steps=3000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=500 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --opt_epsilon=1e-8

# Run evaluation.
python ../eval_classifier.py \
  --batch_size=10 \
  --checkpoint_path=${TRAIN_DIR}/stage_1 \
  --eval_dir=${EVAL_DIR}/stage_1 \
  --dataset_split_name=validation \
  --dataset_dir=${TRAIN_EVAL_TFRECORD_DIR} \
  --model_name=inception_resnet_v2 \


#### stage_2 ####
# Fine-tune all the new layers for 6000 steps.
python ../train_classifier.py \
  --train_dir=${TRAIN_DIR}/stage_2 \
  --dataset_split_name=train \
  --dataset_dir=${TRAIN_EVAL_TFRECORD_DIR} \
  --checkpoint_path=${TRAIN_DIR}/stage_1 \
  --model_name=inception_resnet_v2 \
  --max_number_of_steps=6000 \
  --batch_size=32 \
  --learning_rate=0.00001 \
  --save_interval_secs=1200 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --opt_epsilon=1e-8

# Run evaluation.
python ../eval_classifier.py \
  --batch_size=10 \
  --checkpoint_path=${TRAIN_DIR}/stage_2 \
  --eval_dir=${EVAL_DIR}/stage_2 \
  --dataset_split_name=validation \
  --dataset_dir=${TRAIN_EVAL_TFRECORD_DIR} \
  --model_name=inception_resnet_v2


#### stage_3 ####
# Fine-tune all the new layers for 3000 steps.
python ../train_classifier.py \
  --train_dir=${TRAIN_DIR}/stage_3 \
  --dataset_split_name=train \
  --dataset_dir=${TRAIN_EVAL_TFRECORD_DIR} \
  --checkpoint_path=${TRAIN_DIR}/stage_2 \
  --model_name=inception_resnet_v2 \
  --max_number_of_steps=3000 \
  --batch_size=32 \
  --learning_rate=0.000005 \
  --save_interval_secs=600 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --opt_epsilon=1e-8

# Run evaluation.
python ../eval_classifier.py \
  --batch_size=10 \
  --checkpoint_path=${TRAIN_DIR}/stage_3 \
  --eval_dir=${EVAL_DIR}/stage_3 \
  --dataset_split_name=validation \
  --dataset_dir=${TRAIN_EVAL_TFRECORD_DIR} \
  --model_name=inception_resnet_v2


#### test ####
# Create test_output.
python ../test_classifier.py \
    --checkpoint_path=${TRAIN_DIR}/stage_3 \
    --output_path=${TEST_DIR}/${TEST_OUTPUT} \
    --test_path=${TEST_DATASET_PATH} \
    --num_classes=${NUM_CLASSES} \
    --model_name=inception_resnet_v2

# Create predict_train7000_output.
python ../test_classifier.py \
    --checkpoint_path=${TRAIN_DIR}/stage_3 \
    --output_path=${TEST_DIR}/${TEST_TRAIN_OUTPUT} \
    --test_path=${TRAIN_DATASET_PATH} \
    --test_number=${TEST_TRAIN_NUM} \
    --num_classes=${NUM_CLASSES} \
    --model_name=inception_resnet_v2
