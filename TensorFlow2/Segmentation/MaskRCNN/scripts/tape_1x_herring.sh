#!/usr/bin/env bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -rf $BASEDIR/../results_tape_1x
mkdir -p $BASEDIR/../results_tape_1x
herringrun --singlenode    /shared/conda/bin/python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
        --loop_mode="tape" \
        --checkpoint="/shared//DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=100 \
        --init_learning_rate=0.01 \
        --learning_rate_steps="118280,162635" \
        --optimizer_type="SGD" \
        --lr_schedule="piecewise" \
        --model_dir="$BASEDIR/../results_tape_1x" \
        --num_steps_per_eval=14785 \
        --warmup_learning_rate=0.000133 \
        --warmup_steps=500 \
        --global_gradient_clip_ratio=5.0 \
        --total_steps=192205 \
        --l2_weight_decay=1e-4 \
        --train_batch_size=1 \
        --eval_batch_size=1 \
        --dist_eval \
        --training_file_pattern="/shared/data/nv_tfrecords/train*.tfrecord" \
        --validation_file_pattern="/shared/data/nv_tfrecords/val*.tfrecord" \
        --val_json_file="/shared/data/nv_tfrecords/annotations/instances_val2017.json" \
        --amp \
        --xla \
        --use_batched_nms \
        --async_eval \
        --use_ext \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tape_1x/results_tape_1x.log
