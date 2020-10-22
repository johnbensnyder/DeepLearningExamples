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

BATCH_SIZE=1
HOST_COUNT=1
GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
NUM_GPUS=${NUM_GPUS:-${GPU_COUNT}}
IMAGES=118287
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * HOST_COUNT * GPU_COUNT))
STEP_PER_EPOCH=$(( IMAGES / GLOBAL_BATCH_SIZE ))
FIRST_DECAY=$(( 8 * STEP_PER_EPOCH ))
SECOND_DECAY=$(( 11 * STEP_PER_EPOCH ))
TOTAL_STEPS=${TOTAL_STEPS:-$(( 13 * STEP_PER_EPOCH ))}
DATA_PATH=${DATA_PATH:-"/data/coco/coco-2017"}
LR_MULTIPLIER=0.001
BASE_LR=$(echo $GLOBAL_BATCH_SIZE*$LR_MULTIPLIER | bc)
DIRECT_LAUNCH=${DIRECT_LAUNCH:-"0"}


BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
#PROFILE_PATH="--profile_path ${BASEDIR}/../Profiles/TapeSingleHost"

rm -rf $BASEDIR/../baseline_1x_tape
mkdir -p $BASEDIR/../baseline_1x_tape
/opt/amazon/openmpi/bin/mpirun --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    -np ${NUM_GPUS} -H localhost:${NUM_GPUS} \
    -N 8 \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    --bind-to none \
    /shared/sami/conda/bin/python  ${BASEDIR}/bind_launch.py  --direct_launch=${DIRECT_LAUNCH} --nproc_per_node=${NUM_GPUS} --nsockets_per_node=2 --ncores_per_socket=24 ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train" \
        --eval_after_training=0\
        --loop_mode="tape" \
        --eval_after_training=0 \
        --checkpoint="${BASEDIR}/../resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=1000 \
        --init_learning_rate=$BASE_LR \
        --learning_rate_steps="$FIRST_DECAY,$SECOND_DECAY" \
        --optimizer_type="SGD" \
        --lr_schedule="piecewise" \
        --model_dir="$BASEDIR/../baseline_1x_tape" \
        --num_steps_per_eval=$STEP_PER_EPOCH \
        --warmup_learning_rate=0.000133 \
        --warmup_steps=100 \
        --global_gradient_clip_ratio=0.0 \
        --total_steps=$TOTAL_STEPS \
        --l2_weight_decay=1e-4 \
        --train_batch_size=$BATCH_SIZE \
        --eval_batch_size=1 \
        --dist_eval \
        --training_file_pattern="/home/ubuntu/data2/train*.tfrecord" \
        --validation_file_pattern="/home/ubuntu/data2/val*.tfrecord" \
        --val_json_file="/home/ubuntu2/data2/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --async_eval \
        --use_ext \
        ${PROFILE_PATH} \
        --use_custom_box_proposals_op | tee $BASEDIR/../baseline_1x/baseline_1x.log