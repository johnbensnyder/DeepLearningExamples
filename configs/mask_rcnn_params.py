#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""Parameters used to build Mask-RCNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace
from glob import glob

import tensorflow as tf
import tensorflow_addons as tfa

from awsdet.utils.logging_formatter import logging
from awsdet.utils.dist_utils import MPI_size
from awsdet.training.schedulers import WarmupScheduler

class _Namespace(Namespace):
    def values(self):
        return self.__dict__


def default_config():
    config = _Namespace(**dict(
        # input pre-processing parameters
        f
        augment_input_data=True,
        gt_mask_size=112,

        # dataset specific parameters
        num_classes=91,
        # num_classes=81,
        skip_crowd_during_training=True,
        use_category=True,

        # Region Proposal Network
        rpn_positive_overlap=0.7,
        rpn_negative_overlap=0.3,
        rpn_batch_size_per_im=256,
        rpn_fg_fraction=0.5,
        rpn_min_size=0.,

        # Proposal layer.
        batch_size_per_im=512,
        fg_fraction=0.25,
        fg_thresh=0.5,
        bg_thresh_hi=0.5,
        bg_thresh_lo=0.,

        # Faster-RCNN heads.
        fast_rcnn_mlp_head_dim=1024,
        bbox_reg_weights=(10., 10., 5., 5.),

        # Mask-RCNN heads.
        include_mask=True,  # whether or not to include mask branch.   # ===== Not existing in MLPerf ===== #
        mrcnn_resolution=28,

        # training
        train_rpn_pre_nms_topn=2000,
        train_rpn_post_nms_topn=1000,
        train_rpn_nms_threshold=0.7,

        # evaluation
        test_detections_per_image=100,
        test_nms=0.5,
        test_rpn_pre_nms_topn=1000,
        test_rpn_post_nms_topn=1000,
        test_rpn_nms_thresh=0.7,

        # model architecture
        min_level=2,
        max_level=6,
        num_scales=1,
        aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        anchor_scale=8.0,

        # localization loss
        rpn_box_loss_weight=1.0,
        fast_rcnn_box_loss_weight=1.0,
        mrcnn_weight_loss_mask=1.0,

        # ---------- Training configurations ----------

        # Skips loading variables from the resnet checkpoint. It is used for
        # skipping nonexistent variables from the constructed graph. The list
        # of loaded variables is constructed from the scope 'resnetX', where 'X'
        # is depth of the resnet model. Supports regular expression.
        # skip_checkpoint_variables='^NO_SKIP$',
        skip_checkpoint_variables='^.*(group_norm)',
        train_batch_size_per_gpu=4,
        eval_batch_size_per_gpu=4,
        base_lr=15e-4, # scales linearly with total batch size
        training_file_pattern="/workspace/data/coco/train*",
        validation_file_pattern="/workspace/data/coco/val*",
        validation_annotations="/workspace/data/coco/annotations/instances_val2017.json",
        backbone_checkpoint="/model/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603",
        images=118287, # check this number
        amp=True,
        xla=True,
        dist_eval=True,
        data_slack=False,
        flatten_masks=True,
        use_batched_nms=True,
        use_custom_box_proposals_op=True,
        finetune_bn=False,
        global_gradient_clip_ratio=15.0,
        l2_weight_decay=1e-4,
        warmup_epochs=0.1,
        initial_learning_rate=15e-6,
        lr_schedule="piecewise", # piecewise or cosine supported
        optimizer_type="SGD", # SGD and NovoGrad supported
        box_loss_type="huber", # huber or giou
        momentum=0.9,
        nesterov=False,
        beta1=0.9,
        beta2=0.3,
        learning_rate_decay_levels=[1, 0.1, 0.01],
        learning_rate_decay_epochs=[8, 11],
        num_epochs=12,
        use_ext=True,
        async_eval=False,
        include_groundtruth_in_features=False,

        # ---------- Eval configurations ----------
        # Visualizes images and detection boxes on TensorBoard.
        visualize_images_summary=False,
    ))
    config.global_train_batch_size = MPI_size() * config.train_batch_size_per_gpu
    config.global_eval_batch_size = MPI_size() * config.eval_batch_size_per_gpu
    config.steps_per_epoch = config.images//config.global_train_batch_size
    config.total_steps = config.num_epochs * config.steps_per_epoch
    config.schedule = build_scheduler(config)
    config.optimizer = build_optimizer(config)
    config.training_files = glob(config.training_file_pattern)
    config.validation_files = glob(config.validation_file_pattern)
    return config

def build_scheduler(config):
    config.scaled_lr = config.base_lr * MPI_size()
    if config.lr_schedule=="piecewise":
        boundaries = [config.steps_per_epoch * epoch for epoch in config.learning_rate_decay_epochs]
        lr_steps = [config.scaled_lr * level for level in config.learning_rate_decay_levels]
        schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_steps)
    elif params.lr_schedule=="cosine":
        schedule = tf.keras.experimental.CosineDecay(config.scaled_lr, config.total_steps)
    else:
        raise NotImplementedError
    warmup_steps = int(config.warmup_epochs * config.steps_per_epoch)
    schedule = WarmupScheduler(schedule, config.initial_learning_rate, warmup_steps)
    return schedule

def build_optimizer(config):
    if config.optimizer_type=="SGD":
        optimizer = tf.keras.optimizers.SGD(config.schedule, momentum=config.momentum, nesterov=config.nesterov)
    elif optimizer_type=="NovoGrad":
        config.optimizer = tfa.optimizers.NovoGrad(config.schedule, beta_1=config.beta1, beta_2=config.beta2, weight_decay=config.l2_weight_decay)
    else:
        raise NotImplementedError
    if config.amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    return optimizer
