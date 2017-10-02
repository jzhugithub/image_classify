#!/bin/bash

################
# Before train #
################
# 1. convert dataset
# modify parameters in 'convert_data.py'
# cd to image_classify_ai
# run: python convert_data.py

# 2. set dataset for model
# modify parameter in 'read_dataset.py'

###############
# Start train #
###############
# modify parameters in '{scripts}/script.sh'
# cd to image_classify_ai/scripts
# run: bash {script}.sh

###################
# Run tensorboard #
###################
# tensorboard --logdir=${MODEL_DIR}
# http://localhost:6006


# Where the dataset is saved to.
DATASET_DIR=/home/zj/database_temp/ai_challenger_scene/tfrecord

# Where the pre-trained inception_resnet_v2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_PATH=/home/zj/database_temp/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt

# Where the checkpoint and logs will be saved to.
MODEL_DIR=/home/zj/my_workspace/image_classify/models/resnet_v1_50

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=${MODEL_DIR}/train

# Where the evaluation logs will be saved to.
EVAL_DIR=${MODEL_DIR}/eval

# make dictionary to use.
mkdir -p ${TRAIN_DIR}/stage_1
mkdir -p ${TRAIN_DIR}/stage_2
mkdir -p ${EVAL_DIR}/stage_1
mkdir -p ${EVAL_DIR}/stage_2

# Fine-tune only the new layers for 3000 steps.
python ../train_classifier.py \
  --train_dir=${TRAIN_DIR}/stage_1 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_PATH} \
  --checkpoint_exclude_scopes=resnet_v1_50/logits \
  --trainable_scopes=resnet_v1_50/logits \
  --max_number_of_steps=6000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python ../eval_classifier.py \
  --batch_size=10 \
  --checkpoint_path=${TRAIN_DIR}/stage_1 \
  --eval_dir=${EVAL_DIR}/stage_1 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \

# Fine-tune all the new layers for 1000 steps.
python ../train_classifier.py \
  --train_dir=${TRAIN_DIR}/stage_2 \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --checkpoint_path=${TRAIN_DIR}/stage_1 \
  --model_name=resnet_v1_50 \
  --max_number_of_steps=3000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --save_interval_secs=180 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python ../eval_classifier.py \
  --batch_size=10 \
  --checkpoint_path=${TRAIN_DIR}/stage_2 \
  --eval_dir=${EVAL_DIR}/stage_2 \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50
