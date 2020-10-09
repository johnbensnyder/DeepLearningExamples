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

"""Model definition for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

import itertools
import copy
import numpy as np
import multiprocessing
from statistics import mean
import threading
from math import ceil
from mpi4py import MPI
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.core.protobuf import rewriter_config_pb2
from mask_rcnn import anchors

from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.tf2.models import heads as tf2_heads
from mask_rcnn.models import resnet

from mask_rcnn.training import losses, learning_rates

from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_local_rank, MPI_rank
from mask_rcnn import evaluation, coco_metric

from mask_rcnn.utils.meters import StandardMeter
from mask_rcnn.utils.metric_tracking import register_metric

from mask_rcnn.utils.lazy_imports import LazyImport
from mask_rcnn.training.optimization import LambOptimizer, NovoGrad
from mask_rcnn.tf2.utils import warmup_scheduler, eager_mapping

from mask_rcnn.utils.meters import StandardMeter
from mask_rcnn.utils.metric_tracking import register_metric

hvd = LazyImport("horovod.tensorflow")

class MaskRCNN_Parallel(tf.keras.Model):
    
    def __init__(self, params, devices, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.devices = devices
        with tf.device(self.devices[0].name):
            self.backbone_0 = resnet.Resnet_Model("resnet50",
                                     data_format='channels_last',
                                     trainable=trainable,
                                     finetune_bn=params['finetune_bn']
                                     )
            self.fpn_0 = fpn.FPNNetwork(params['min_level'], params['max_level'], trainable=trainable)
            num_anchors=len(params['aspect_ratios'] * params['num_scales'])
            self.rpn_0 = heads.RPN_Head_Model(name="rpn_head_0", 
                                            num_anchors=num_anchors, trainable=trainable)
            is_gpu_inference = not trainable and params['use_batched_nms']
            self.box_head_0 = heads.Box_Head_Model(num_classes=params['num_classes'],
                                                 mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
                                                 trainable=trainable
                                                )
            if params['include_mask']:
                self.mask_head_0 = tf2_heads.Mask_Head_Model(num_classes=params['num_classes'],
                                                           mrcnn_resolution=params['mrcnn_resolution'],
                                                           is_gpu_inference=is_gpu_inference,
                                                           trainable=trainable,
                                                           name="mask_head_0"
                                                           )
        with tf.device(self.devices[1].name):
            self.backbone_1 = resnet.Resnet_Model("resnet50",
                                     data_format='channels_last',
                                     trainable=trainable,
                                     finetune_bn=params['finetune_bn']
                                     )
            self.fpn_1 = fpn.FPNNetwork(params['min_level'], params['max_level'], trainable=trainable)
            num_anchors=len(params['aspect_ratios'] * params['num_scales'])
            self.rpn_1 = heads.RPN_Head_Model(name="rpn_head_1", 
                                            num_anchors=num_anchors, trainable=trainable)
            is_gpu_inference = not trainable and params['use_batched_nms']
            self.box_head_1 = heads.Box_Head_Model(num_classes=params['num_classes'],
                                                 mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
                                                 trainable=trainable
                                                )
            if params['include_mask']:
                self.mask_head_1 = tf2_heads.Mask_Head_Model(num_classes=params['num_classes'],
                                                           mrcnn_resolution=params['mrcnn_resolution'],
                                                           is_gpu_inference=is_gpu_inference,
                                                           trainable=trainable,
                                                           name="mask_head_1"
                                                           )
        for key, value in params.items():
            self.__dict__[key] = value
    
    def call(self, features_0, features_1, labels_0=None, labels_1=None, training=True):
        model_outputs = {}
        is_gpu_inference = not training and self.use_batched_nms
        # split images and put on both GPUs
        all_anchors = anchors.Anchors(self.min_level, self.max_level,
                                  self.num_scales, self.aspect_ratios,
                                  self.anchor_scale,
                                  (832, 1344))
        with tf.device(self.devices[0].name):
            img_shape_0 = tf.cast(tf.shape(features_0['images']), tf.int32)
            images_0 = features_0['images'][:,:,:672+64,:]
        with tf.device(self.devices[1].name):
            img_shape_1 = tf.cast(tf.shape(features_1['images']), tf.int32)
            images_1 = features_1['images'][:,:,672-64:,:]
        # run backbone on each GPU
        with tf.device(self.devices[0].name):
            res_out_0 = self.backbone_0(images_0, training=training)
            res_out_0[2] = res_out_0[2][:,:,:-16,:]
            res_out_0[3] = res_out_0[3][:,:,:-8,:]
            res_out_0[4] = res_out_0[4][:,:,:-4,:]
            res_out_0[5] = res_out_0[5][:,:,:-2,:]
        with tf.device(self.devices[1].name):
            res_out_1 = self.backbone_1(images_1, training=training)
            res_out_1[2] = res_out_1[2][:,:,16:,:]
            res_out_1[3] = res_out_1[3][:,:,8:,:]
            res_out_1[4] = res_out_1[4][:,:,4:,:]
            res_out_1[5] = res_out_1[5][:,:,2:,:]
        # copy tensors from device 1 to device 0 cut off overlapping section
        with tf.device(self.devices[0].name):
            C2_1_0 = tf.stop_gradient(tf.identity(res_out_1[2]))
            C3_1_0 = tf.stop_gradient(tf.identity(res_out_1[3]))
            C4_1_0 = tf.stop_gradient(tf.identity(res_out_1[4]))
            C5_1_0 = tf.stop_gradient(tf.identity(res_out_1[5]))

        # copy tensors from device 0 to device 1 cut off overlapping section
        with tf.device(self.devices[1].name):
            C2_0_1 = tf.stop_gradient(tf.identity(res_out_0[2]))
            C3_0_1 = tf.stop_gradient(tf.identity(res_out_0[3]))
            C4_0_1 = tf.stop_gradient(tf.identity(res_out_0[4]))
            C5_0_1 = tf.stop_gradient(tf.identity(res_out_0[5]))

        # concatenate tensors
        with tf.device(self.devices[0].name):
            res_out_0[2] = tf.concat([res_out_0[2], C2_1_0], axis=2)
            res_out_0[3] = tf.concat([res_out_0[3], C3_1_0], axis=2)
            res_out_0[4] = tf.concat([res_out_0[4], C4_1_0], axis=2)
            res_out_0[5] = tf.concat([res_out_0[5], C5_1_0], axis=2)

        with tf.device(self.devices[1].name):
            res_out_1[2] = tf.concat([C2_0_1, res_out_1[2]], axis=2)
            res_out_1[3] = tf.concat([C3_0_1, res_out_1[3]], axis=2)
            res_out_1[4] = tf.concat([C4_0_1, res_out_1[4]], axis=2)
            res_out_1[5] = tf.concat([C5_0_1, res_out_1[5]], axis=2)
        
        # run full FPN and RPN on both GPUs
        with tf.device(self.devices[0].name):
            fpn_feats_0 = self.fpn_0(res_out_0, training=training)
            rpn_score_outputs_0 = dict()
            rpn_box_outputs_0 = dict()
            for level in range(self.min_level, self.max_level + 1):
                rpn_score_outputs_0[level], rpn_box_outputs_0[level] = \
                    self.rpn_0(fpn_feats_0[level], training=training)
                
        with tf.device(self.devices[1].name):
            fpn_feats_1 = self.fpn_1(res_out_1, training=training)
            rpn_score_outputs_1 = dict()
            rpn_box_outputs_1 = dict()
            for level in range(self.min_level, self.max_level + 1):
                rpn_score_outputs_1[level], rpn_box_outputs_1[level] = \
                    self.rpn_1(fpn_feats_1[level], training=training)
                
        # run NMS on both GPUs
        if training:
            rpn_pre_nms_topn = self.train_rpn_pre_nms_topn
            rpn_post_nms_topn = self.train_rpn_post_nms_topn
            rpn_nms_threshold = self.train_rpn_nms_threshold
        else:
            rpn_pre_nms_topn = self.test_rpn_pre_nms_topn
            rpn_post_nms_topn = self.test_rpn_post_nms_topn
            rpn_nms_threshold = self.test_rpn_nms_thresh

        with tf.device(self.devices[0].name):
            rpn_box_scores_0, rpn_box_rois_0 = roi_ops.custom_multilevel_propose_rois(
                scores_outputs=rpn_score_outputs_0,
                box_outputs=rpn_box_outputs_0,
                all_anchors=all_anchors,
                image_info=features_0['image_info'],
                rpn_pre_nms_topn=rpn_pre_nms_topn,
                rpn_post_nms_topn=rpn_post_nms_topn,
                rpn_nms_threshold=rpn_nms_threshold,
                rpn_min_size=self.rpn_min_size
            )
            rpn_box_rois_0 = tf.cast(rpn_box_rois_0, dtype=tf.float32)
            if training:
                rpn_box_rois_0 = tf.stop_gradient(rpn_box_rois_0)
                rpn_box_scores_0 = tf.stop_gradient(rpn_box_scores_0)
        with tf.device(self.devices[1].name):
            rpn_box_scores_1, rpn_box_rois_1 = roi_ops.custom_multilevel_propose_rois(
                scores_outputs=rpn_score_outputs_1,
                box_outputs=rpn_box_outputs_1,
                all_anchors=all_anchors,
                image_info=features_1['image_info'],
                rpn_pre_nms_topn=rpn_pre_nms_topn,
                rpn_post_nms_topn=rpn_post_nms_topn,
                rpn_nms_threshold=rpn_nms_threshold,
                rpn_min_size=self.rpn_min_size
            )
            rpn_box_rois_1 = tf.cast(rpn_box_rois_1, dtype=tf.float32)
            if training:
                rpn_box_rois_1 = tf.stop_gradient(rpn_box_rois_1)
                rpn_box_scores_1 = tf.stop_gradient(rpn_box_scores_1)
        
        if training:
            # run sampling without device placement
            box_targets_0, class_targets_0, \
            rpn_box_rois_0, proposal_to_label_map_0 = training_ops.proposal_label_op(
                    rpn_box_rois_0,
                    labels_0['gt_boxes'],
                    labels_0['gt_classes'],
                    batch_size_per_im=self.batch_size_per_im,
                    fg_fraction=self.fg_fraction,
                    fg_thresh=self.fg_thresh,
                    bg_thresh_hi=self.bg_thresh_hi,
                    bg_thresh_lo=self.bg_thresh_lo
                )
            box_targets_1, class_targets_1, \
            rpn_box_rois_1, proposal_to_label_map_1 = training_ops.proposal_label_op(
                    rpn_box_rois_1,
                    labels_1['gt_boxes'],
                    labels_1['gt_classes'],
                    batch_size_per_im=self.batch_size_per_im,
                    fg_fraction=self.fg_fraction,
                    fg_thresh=self.fg_thresh,
                    bg_thresh_hi=self.bg_thresh_hi,
                    bg_thresh_lo=self.bg_thresh_lo
                )
        # Run both heads on first GPU
        with tf.device(self.devices[0].name):
            # Performs multi-level RoIAlign.
            box_roi_features_0 = spatial_transform_ops.multilevel_crop_and_resize(
                features=fpn_feats_0,
                boxes=rpn_box_rois_0,
                output_size=7,
                is_gpu_inference=is_gpu_inference
            )
            class_outputs_0, box_outputs_0, _ = self.box_head_0(inputs=box_roi_features_0)
            if not training:
                generate_detections_fn = postprocess_ops.generate_detections_tpu
                detections_0 = generate_detections_fn(
                    class_outputs=class_outputs_0,
                    box_outputs=box_outputs_0,
                    anchor_boxes=rpn_box_rois_0,
                    image_info=features_0['image_info'],
                    pre_nms_num_detections=self.test_rpn_post_nms_topn,
                    post_nms_num_detections=self.test_detections_per_image,
                    nms_threshold=self.test_nms,
                    bbox_reg_weights=self.bbox_reg_weights
                )
                model_outputs.update({
                    'num_detections_0': detections_0[0],
                    'detection_boxes_0': detections_0[1],
                    'detection_classes_0': detections_0[2],
                    'detection_scores_0': detections_0[3],
                })
            else:  # is training
                encoded_box_targets_0 = training_ops.encode_box_targets(
                    boxes=rpn_box_rois_0,
                    gt_boxes=box_targets_0,
                    gt_labels=class_targets_0,
                    bbox_reg_weights=self.bbox_reg_weights
                )
                model_outputs.update({
                'rpn_score_outputs_0': rpn_score_outputs_0,
                'rpn_box_outputs_0': rpn_box_outputs_0,
                'class_outputs_0': tf.cast(class_outputs_0, tf.float32),
                'box_outputs_0': tf.cast(box_outputs_0, tf.float32),
                'class_targets_0': class_targets_0,
                'box_targets_0': encoded_box_targets_0,
                'box_rois_0': rpn_box_rois_0,
                })
            if not training:
                selected_box_rois_0 = model_outputs['detection_boxes_0']
                class_indices_0 = model_outputs['detection_classes_0']
                class_indices_0 = tf.cast(class_indices_0, dtype=tf.int32)
            else:
                selected_class_targets_0, selected_box_targets_0, \
                selected_box_rois_0, proposal_to_label_map_0 = \
                training_ops.select_fg_for_masks(
                    class_targets=class_targets_0,
                    box_targets=box_targets_0,
                    boxes=rpn_box_rois_0,
                    proposal_to_label_map=proposal_to_label_map_0,
                    max_num_fg=int(self.batch_size_per_im * self.fg_fraction)
                )
                class_indices_0 = tf.cast(selected_class_targets_0, dtype=tf.int32)
            mask_roi_features_0 = spatial_transform_ops.multilevel_crop_and_resize(
                features=fpn_feats_0,
                boxes=selected_box_rois_0,
                output_size=14,
                is_gpu_inference=is_gpu_inference
            )
            mask_outputs_0 = self.mask_head_0(inputs=mask_roi_features_0, 
                                              class_indices=class_indices_0)
            if training:
                mask_targets_0 = training_ops.get_mask_targets(

                    fg_boxes=selected_box_rois_0,
                    fg_proposal_to_label_map=proposal_to_label_map_0,
                    fg_box_targets=selected_box_targets_0,
                    mask_gt_labels=labels_0['cropped_gt_masks'],
                    output_size=self.mrcnn_resolution
                )

                model_outputs.update({
                    'mask_outputs_0': tf.cast(mask_outputs_0, tf.float32),
                    'mask_targets_0': mask_targets_0,
                    'selected_class_targets_0': selected_class_targets_0,
                })

            else:
                model_outputs.update({
                    'detection_masks_0': tf.nn.sigmoid(mask_outputs_0),
                })
        
        # Run both heads on second GPU
        with tf.device(self.devices[1].name):
            # Performs multi-level RoIAlign.
            box_roi_features_1 = spatial_transform_ops.multilevel_crop_and_resize(
                features=fpn_feats_1,
                boxes=rpn_box_rois_1,
                output_size=7,
                is_gpu_inference=is_gpu_inference
            )
            class_outputs_1, box_outputs_1, _ = self.box_head_1(inputs=box_roi_features_1)
            if not training:
                generate_detections_fn = postprocess_ops.generate_detections_tpu
                detections_1 = generate_detections_fn(
                    class_outputs=class_outputs_1,
                    box_outputs=box_outputs_1,
                    anchor_boxes=rpn_box_rois_1,
                    image_info=features_1['image_info'],
                    pre_nms_num_detections=self.test_rpn_post_nms_topn,
                    post_nms_num_detections=self.test_detections_per_image,
                    nms_threshold=self.test_nms,
                    bbox_reg_weights=self.bbox_reg_weights
                )
                model_outputs.update({
                    'num_detections_1': detections_1[0],
                    'detection_boxes_1': detections_1[1],
                    'detection_classes_1': detections_1[2],
                    'detection_scores_1': detections_1[3],
                })
            else:  # is training
                encoded_box_targets_1 = training_ops.encode_box_targets(
                    boxes=rpn_box_rois_1,
                    gt_boxes=box_targets_1,
                    gt_labels=class_targets_1,
                    bbox_reg_weights=self.bbox_reg_weights
                )
                model_outputs.update({
                'rpn_score_outputs_1': rpn_score_outputs_1,
                'rpn_box_outputs_1': rpn_box_outputs_1,
                'class_outputs_1': tf.cast(class_outputs_1, tf.float32),
                'box_outputs_1': tf.cast(box_outputs_1, tf.float32),
                'class_targets_1': class_targets_1,
                'box_targets_1': encoded_box_targets_1,
                'box_rois_1': rpn_box_rois_1,
                })
            if not training:
                selected_box_rois_1 = model_outputs['detection_boxes_1']
                class_indices_1 = model_outputs['detection_classes_1']
                class_indices_1 = tf.cast(class_indices_1, dtype=tf.int32)
            else:
                selected_class_targets_1, selected_box_targets_1, \
                selected_box_rois_1, proposal_to_label_map_1 = \
                training_ops.select_fg_for_masks(
                    class_targets=class_targets_1,
                    box_targets=box_targets_1,
                    boxes=rpn_box_rois_1,
                    proposal_to_label_map=proposal_to_label_map_1,
                    max_num_fg=int(self.batch_size_per_im * self.fg_fraction)
                )
                class_indices_1 = tf.cast(selected_class_targets_1, dtype=tf.int32)
            mask_roi_features_1 = spatial_transform_ops.multilevel_crop_and_resize(
                features=fpn_feats_1,
                boxes=selected_box_rois_1,
                output_size=14,
                is_gpu_inference=is_gpu_inference
            )
            mask_outputs_1 = self.mask_head_1(inputs=mask_roi_features_1, 
                                              class_indices=class_indices_1)
            if training:
                mask_targets_1 = training_ops.get_mask_targets(

                    fg_boxes=selected_box_rois_1,
                    fg_proposal_to_label_map=proposal_to_label_map_1,
                    fg_box_targets=selected_box_targets_1,
                    mask_gt_labels=labels_1['cropped_gt_masks'],
                    output_size=self.mrcnn_resolution
                )

                model_outputs.update({
                    'mask_outputs_1': tf.cast(mask_outputs_1, tf.float32),
                    'mask_targets_1': mask_targets_1,
                    'selected_class_targets_1': selected_class_targets_1,
                })

            else:
                model_outputs.update({
                    'detection_masks_1': tf.nn.sigmoid(mask_outputs_1),
                })
            
        return model_outputs
    
class TapeModel(object):
    def __init__(self, params, devices, train_input_fn=None, eval_input_fn=None, is_training=True):
        self.params = params
        self.devices = devices
        self.forward = MaskRCNN_Parallel(self.params.values(), devices, trainable=is_training)
        train_params = dict(self.params.values(), batch_size=self.params.train_batch_size)
        self.train_tdf_0 = iter(train_input_fn(train_params).apply(tf.data.experimental.copy_to_device(devices[0].name))) \
                            if train_input_fn else None
        self.train_tdf_1 = iter(train_input_fn(train_params).apply(tf.data.experimental.copy_to_device(devices[1].name))) \
                            if train_input_fn else None
        with tf.device(self.devices[0].name):
            self.optimizer_0, self.schedule_0 = self.get_optimizer()
        with tf.device(self.devices[1].name):
            self.optimizer_1, self.schedule_1 = self.get_optimizer()
    
    def load_weights(self):
        chkp = tf.compat.v1.train.NewCheckpointReader(self.params.checkpoint)
        weights = [chkp.get_tensor(i) for i in eager_mapping.resnet_vars]
        self.forward.layers[0].set_weights(weights)
        self.forward.layers[5].set_weights(weights)
    
    def get_optimizer(self):
        if self.params.lr_schedule=='piecewise':
            schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.params.learning_rate_steps,
                                                                            [self.params.init_learning_rate] + \
                                                                            self.params.learning_rate_levels)
        else:
            raise NotImplementedError
        schedule = warmup_scheduler.WarmupScheduler(schedule, self.params.warmup_learning_rate,
                                                    self.params.warmup_steps)
        if self.params.optimizer_type=="SGD":
            opt = tf.keras.optimizers.SGD(learning_rate=schedule, 
                                          momentum=self.params.momentum)
        elif self.params.optimizer_type=="LAMB":
            opt = tfa.optimizers.LAMB(learning_rate=schedule)
        elif self.params.optimizer_type=="Novograd":
            opt = tfa.optimizers.NovoGrad(learning_rate=schedule)
        else:
            raise NotImplementedError
        if self.params.amp:
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        return opt, schedule
    
    def initialize_model(self):
        features_0, labels_0 = next(self.train_tdf_0)
        features_1, labels_1 = next(self.train_tdf_1)
        model_outputs = self.forward(features_0, features_1, labels_0, labels_1, True)
        self.load_weights()
    
    @tf.function
    def train_step(self, features_0, features_1, labels_0, labels_1, sync_weights=False, sync_opt=False):
        loss_dict = dict()
        #with tf.GradientTape() as tape_0, tf.GradientTape() as tape_1:
        with tf.GradientTape(persistent=True) as tape:
            model_outputs = self.forward(features_0, features_1, labels_0, labels_1, True)
            with tf.device(self.devices[0].name):
                loss_dict['total_rpn_loss_0'], loss_dict['rpn_score_loss_0'], \
                    loss_dict['rpn_box_loss_0'] = losses.rpn_loss(
                    score_outputs=model_outputs['rpn_score_outputs_0'],
                    box_outputs=model_outputs['rpn_box_outputs_0'],
                    labels=labels_0,
                    params=self.params.values()
                )
                loss_dict['total_fast_rcnn_loss_0'], loss_dict['fast_rcnn_class_loss_0'], \
                    loss_dict['fast_rcnn_box_loss_0'] = losses.fast_rcnn_loss(
                    class_outputs=model_outputs['class_outputs_0'],
                    box_outputs=model_outputs['box_outputs_0'],
                    class_targets=model_outputs['class_targets_0'],
                    box_targets=model_outputs['box_targets_0'],
                    params=self.params.values()
                )
                if self.params.include_mask:
                    loss_dict['mask_loss_0'] = losses.mask_rcnn_loss(
                        mask_outputs=model_outputs['mask_outputs_0'],
                        mask_targets=model_outputs['mask_targets_0'],
                        select_class_targets=model_outputs['selected_class_targets_0'],
                        params=self.params.values()
                    )
                else:
                    loss_dict['mask_loss_0'] = 0.
                if self.params.optimizer_type in ['LAMB', 'Novograd']: # decoupled weight decay
                    loss_dict['l2_regularization_loss_0'] = tf.constant(0.0)
                else:
                    loss_dict['l2_regularization_loss_0'] = self.params.l2_weight_decay * tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in self.forward.trainable_variables
                        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
                    ])
                loss_dict['total_loss_0'] = loss_dict['total_rpn_loss_0'] \
                    + loss_dict['total_fast_rcnn_loss_0'] + loss_dict['mask_loss_0'] \
                    + loss_dict['l2_regularization_loss_0']
            
            with tf.device(self.devices[1].name):
                loss_dict['total_rpn_loss_1'], loss_dict['rpn_score_loss_1'], \
                    loss_dict['rpn_box_loss_1'] = losses.rpn_loss(
                    score_outputs=model_outputs['rpn_score_outputs_1'],
                    box_outputs=model_outputs['rpn_box_outputs_1'],
                    labels=labels_1,
                    params=self.params.values()
                )
                loss_dict['total_fast_rcnn_loss_1'], loss_dict['fast_rcnn_class_loss_1'], \
                    loss_dict['fast_rcnn_box_loss_1'] = losses.fast_rcnn_loss(
                    class_outputs=model_outputs['class_outputs_1'],
                    box_outputs=model_outputs['box_outputs_1'],
                    class_targets=model_outputs['class_targets_1'],
                    box_targets=model_outputs['box_targets_1'],
                    params=self.params.values()
                )
                if self.params.include_mask:
                    loss_dict['mask_loss_1'] = losses.mask_rcnn_loss(
                        mask_outputs=model_outputs['mask_outputs_1'],
                        mask_targets=model_outputs['mask_targets_1'],
                        select_class_targets=model_outputs['selected_class_targets_1'],
                        params=self.params.values()
                    )
                else:
                    loss_dict['mask_loss_1'] = 0.
                if self.params.optimizer_type in ['LAMB', 'Novograd']: # decoupled weight decay
                    loss_dict['l2_regularization_loss_1'] = tf.constant(0.0)
                else:
                    loss_dict['l2_regularization_loss_1'] = self.params.l2_weight_decay * tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in self.forward.trainable_variables
                        if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
                    ])
                loss_dict['total_loss_1'] = loss_dict['total_rpn_loss_1'] \
                    + loss_dict['total_fast_rcnn_loss_1'] + loss_dict['mask_loss_1'] \
                    + loss_dict['l2_regularization_loss_1']
            loss_dict['total_loss'] = (loss_dict['total_loss_0'] + loss_dict['total_loss_1'])/2
            if self.params.amp:
                scaled_loss_0 = self.optimizer_0.get_scaled_loss(loss_dict['total_loss'])
                scaled_loss_1 = self.optimizer_1.get_scaled_loss(loss_dict['total_loss'])
                #scaled_loss = self.optimizer_0.get_scaled_loss(loss_dict['total_loss'])
        gpu_0_vars = [i for i in self.forward.trainable_variables if 'GPU:0' in i.device]
        gpu_1_vars = [i for i in self.forward.trainable_variables if 'GPU:1' in i.device]
        if MPI_is_distributed():
            #tape_0 = hvd.DistributedGradientTape(tape_0, compression=hvd.compression.NoneCompressor)
            #tape_1 = hvd.DistributedGradientTape(tape_1, compression=hvd.compression.NoneCompressor)
            tape = hvd.DistributedGradientTape(tape, compression=hvd.compression.NoneCompressor)
        if self.params.amp:
            scaled_gradients_0 = tape.gradient(scaled_loss_0, gpu_0_vars)
            gradients_0 = self.optimizer_0.get_unscaled_gradients(scaled_gradients_0)
            scaled_gradients_1 = tape.gradient(scaled_loss_1, gpu_1_vars)
            gradients_1 = self.optimizer_1.get_unscaled_gradients(scaled_gradients_1)
            '''scaled_gradients = tape.gradient(scaled_loss, self.forward.trainable_variables)
            gradients = self.optimizer_0.get_unscaled_gradients(scaled_gradients)'''
        else:
            gradients_0 = tape_0.gradient(loss_dict['total_loss'], gpu_0_vars)
            gradients_1 = tape_1.gradient(loss_dict['total_loss'], gpu_1_vars)
            #gradients = tape.gradient(loss_dict['total_loss'], self.forward.trainable_variables)
        #gradients = [(i+j)/2 for i,j in zip(gradients_0, gradients_1)]
        self.optimizer_0.apply_gradients(zip(gradients_0, gpu_0_vars))
        self.optimizer_1.apply_gradients(zip(gradients_1s, gpu_1_vars))
        if MPI_is_distributed() and sync_weights:
            if MPI_rank()==0:
                logging.info("Broadcasting variables")
            hvd.broadcast_variables(self.forward.variables, 0)
        if MPI_is_distributed() and sync_opt:
            if MPI_rank()==0:
                logging.info("Broadcasting optimizer")
            hvd.broadcast_variables(self.optimizer_0.variables(), 0)
            hvd.broadcast_variables(self.optimizer_1.variables(), 0)
        return loss_dict
    
    def train_epoch(self, steps, broadcast=False):
        if MPI_rank()==0:
            logging.info("Starting training loop")
            p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            loss_history = []
        else:
            p_bar = range(steps)
        for i in p_bar:
            if broadcast and i==0:
                b_w, b_o = True, True
            elif i==0:
                b_w, b_o = False, True
            else:
                b_w, b_o = False, False
            features_0, labels_0 = next(self.train_tdf_0)
            features_1, labels_1 = next(self.train_tdf_1)
            loss_dict = self.train_step(features_0, features_1, labels_0, labels_1, b_w, b_o)
            if MPI_rank()==0:
                loss_history.append(loss_dict['total_loss'].numpy())
                step = self.optimizer_0.iterations
                learning_rate = self.schedule_0(step)
                p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(mean(loss_history[-50:]), 
                                                                          learning_rate))