#!/bin/bash


BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -rf $BASEDIR/../results_tf2_32x_novo_$1
mkdir -p $BASEDIR/../results_tf2_32x_novo_$1

/shared/rejin/conda/bin/herringrun -n 32 --homogeneous -c /shared/rejin/conda \
    RUN_HERRING=1 \
/shared/rejin/conda/bin/python ${BASEDIR}/../mask_rcnn_main.py \
        --mode="train_and_eval" \
	--loop_mode="tape" \
	--box_loss_type="giou" \
        --checkpoint="/shared/rejin/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN//resnet/resnet-nhwc-2018-02-07/model.ckpt-112603" \
        --eval_samples=5000 \
        --log_interval=10 \
        --init_learning_rate=0.05 \
        --optimizer_type="Novograd" \
        --lr_schedule="cosine" \
        --model_dir="$BASEDIR/../results_tf2_32x_novo_$1" \
        --num_steps_per_eval=462 \
        --warmup_learning_rate=0.000133 \
	--beta1=0.9 \
	--beta2=0.5 \
	--warmup_steps=300 \
        --total_steps=7500 \
        --l2_weight_decay=1e-3 \
        --train_batch_size=1 \
        --eval_batch_size=1 \
        --dist_eval \
	--first_eval=15 \
        --training_file_pattern="/shared/rejin/data/nv_tfrecords/train*.tfrecord" \
        --validation_file_pattern="/shared/rejin/data/nv_tfrecords/val*.tfrecord" \
        --val_json_file="/shared/rejin/data/nv_tfrecords/annotations/instances_val2017.json" \
        --amp \
        --use_batched_nms \
        --xla \
        --tf2 \
        --use_custom_box_proposals_op | tee $BASEDIR/../results_tf2_32x_novo_$1/train_eval.log
