#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

/shared/sami/conda/bin/python  ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
        --loop_mode="tape" \
        --box_loss_type="giou" \
        --checkpoint="/shared/sami/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=10 \
        --init_learning_rate=0.05 \
        --optimizer_type="Novograd" \
        --lr_schedule="cosine" \
        --model_dir="$BASEDIR/../results_tf2_32x_novo_$1" \
        --num_steps_per_eval=6000 \
        --warmup_learning_rate=0.000133 \
        --beta1=0.9 \
        --beta2=0.5 \
        --warmup_steps=300 \
        --total_steps=4000 \
        --l2_weight_decay=1e-3 \
        --train_batch_size=1 \
        --eval_batch_size=1 \
        --dist_eval \
        --first_eval=15 \
        --training_file_pattern="/home/ubuntu/data2/train*.tfrecord" \
        --validation_file_pattern="/home/ubuntu/data2/val*.tfrecord" \
        --val_json_file="/home/ubuntu/data2/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tf2_32x_novo_$1/train_eval.log

