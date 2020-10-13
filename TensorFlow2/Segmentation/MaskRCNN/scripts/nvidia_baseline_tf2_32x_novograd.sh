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
<<<<<<< Updated upstream:TensorFlow2/Segmentation/MaskRCNN/scripts/nvidia_baseline_tf2_32x_novograd.sh
rm -rf $BASEDIR/../results_tf2_64x_novo_$1
mkdir -p $BASEDIR/../results_tf2_64x_novo_$1
/shared/rejin/conda/bin/herringrun -n 32 --homogeneous -c /shared/rejin/conda \
    RUN_HERRING=1 \
    /shared/rejin/conda/bin/python ${BASEDIR}/../mask_rcnn_main.py \
=======
rm -rf $BASEDIR/../results_tape_1x
mkdir -p $BASEDIR/../results_tape_1x
/shared/rejin/conda/bin/herringrun -n 64 --homogeneous -c /shared/rejin/conda/ /shared/rejin/conda/bin/python ${BASEDIR}/../mask_rcnn_main.py \
>>>>>>> Stashed changes:TensorFlow2/Segmentation/MaskRCNN/scripts/tape_8x_herring.sh
        --mode="train_and_eval" \
	--loop_mode="tape" \
        --checkpoint="/shared/rejin/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=10 \
        --init_learning_rate=0.08 \
        --optimizer_type="Novograd" \
        --lr_schedule="cosine" \
        --model_dir="$BASEDIR/../results_tf2_64x_novo_$1" \
        --num_steps_per_eval=231 \
        --warmup_learning_rate=0.000133 \
	--beta1=0.9 \
	--beta2=0.25 \
	--warmup_steps=1000 \
        --total_steps=5082 \
        --l2_weight_decay=1.25e-3 \
        --train_batch_size=1 \
        --eval_batch_size=1 \
        --dist_eval \
<<<<<<< Updated upstream:TensorFlow2/Segmentation/MaskRCNN/scripts/nvidia_baseline_tf2_32x_novograd.sh
	--first_eval=15 \
        --training_file_pattern="/shared/rejin/data/nv_tfrecords/train*.tfrecord" \
        --validation_file_pattern="/shared/rejin/data/nv_tfrecords/val*.tfrecord" \
=======
        --training_file_pattern="/shared/data2/train*.tfrecord" \
        --validation_file_pattern="/shared/data2/val*.tfrecord" \
>>>>>>> Stashed changes:TensorFlow2/Segmentation/MaskRCNN/scripts/tape_8x_herring.sh
        --val_json_file="/shared/rejin/data/nv_tfrecords/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tf2_64x_novo_$1/train_eval.log