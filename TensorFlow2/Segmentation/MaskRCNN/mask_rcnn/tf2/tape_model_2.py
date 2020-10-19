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

import time
import itertools
import copy
import numpy as np
import multiprocessing
from statistics import mean
import threading
from math import ceil
from mpi4py import MPI
from tqdm import tqdm
import os

import h5py
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.core.protobuf import rewriter_config_pb2
from mask_rcnn import anchors

from mask_rcnn.models import fpn
from mask_rcnn.models import heads
from mask_rcnn.tf2.models import heads as tf2_heads
from mask_rcnn.models import resnet

from mask_rcnn.training import losses, learning_rates, optimizers

from mask_rcnn.ops import postprocess_ops
from mask_rcnn.ops import roi_ops
from mask_rcnn.ops import spatial_transform_ops
from mask_rcnn.ops import training_ops
from mask_rcnn.ops import preprocess_ops

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
from mask_rcnn.utils.herring_env import is_herring

feature_spec = {'source_ids': tf.TensorSpec(shape=(None), dtype=tf.int64),
                                  'images': tf.TensorSpec(shape=(None, 832, 1344, 3), dtype=tf.float32),
                                  'image_info': tf.TensorSpec(shape=(None, 5), dtype=tf.float32)}
label_spec = {'cropped_gt_masks': tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                               'score_targets_2': tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32),
                               'score_targets_3': tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32),
                               'score_targets_4': tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32),
                               'score_targets_5': tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32),
                               'score_targets_6': tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.int32),
                               'box_targets_2': tf.TensorSpec(shape=(None, None, None, 12), dtype=tf.float32),
                               'box_targets_3': tf.TensorSpec(shape=(None, None, None, 12), dtype=tf.float32),
                               'box_targets_4': tf.TensorSpec(shape=(None, None, None, 12), dtype=tf.float32),
                               'box_targets_5': tf.TensorSpec(shape=(None, None, None, 12), dtype=tf.float32),
                               'box_targets_6': tf.TensorSpec(shape=(None, None, None, 12), dtype=tf.float32),
                               'gt_boxes': tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
                               'gt_classes': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)}


if is_herring():
    import herring.tensorflow as herring
else:
    hvd = LazyImport("horovod.tensorflow")
    
class MRCNN(tf.keras.Model):
    
    def __init__(self, params, is_training=True, **kwargs):
        super().__init__(**kwargs)
        is_gpu_inference = not is_training and params['use_batched_nms']
        self.backbone = resnet.Resnet_Model(
                                        "resnet50",
                                        data_format='channels_last',
                                        trainable=is_training,
                                        finetune_bn=params['finetune_bn']
                                    )
        self.fpn = fpn.FPNNetwork(params['min_level'], 
                                  params['max_level'], 
                                  trainable=is_training)
        self.rpn = heads.RPN_Head_Model(name="rpn_head", 
                                        num_anchors=len(params['aspect_ratios'] * params['num_scales']), 
                                        trainable=is_training)
        self.box_head = heads.Box_Head_Model(
                                    num_classes=params['num_classes'],
                                    mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
                                    trainable=is_training
                                )
        self.mask_head = tf2_heads.Mask_Head_Model(
                                                num_classes=params['num_classes'],
                                                mrcnn_resolution=params['mrcnn_resolution'],
                                                is_gpu_inference=is_gpu_inference,
                                                trainable=is_training,
                                                name="mask_head"
                                            ) 
    
    @tf.function
    def call(self, features, labels, params, is_training=True):
        model_outputs = {}
        is_gpu_inference = not is_training and params['use_batched_nms']
        batch_size, image_height, image_width, _ = features['images'].get_shape().as_list()
        if 'source_ids' not in features:
            features['source_ids'] = -1 * tf.ones([batch_size], dtype=tf.float32)
        rpn_score_outputs, rpn_box_outputs, fpn_feats = self.feature_generator(features['images'],
                                                                    params,
                                                                    is_training=is_training)
        if is_training:
            model_outputs.update({
                'rpn_score_outputs': rpn_score_outputs,
                'rpn_box_outputs': rpn_box_outputs})
        box_targets, class_targets, rpn_box_rois, proposal_to_label_map = \
            self.roi_generator(rpn_score_outputs, rpn_box_outputs, params, 
                               image_height, image_width, features['image_info'], 
                               labels, is_training=is_training)
        if is_training:
            model_outputs.update({
                'class_targets': class_targets,
                'box_rois': rpn_box_rois,
                'box_targets': box_targets
            })
        bbox_outputs = self.box_roi_head(fpn_feats, rpn_box_rois, box_targets, class_targets, params, 
                                          features['image_info'], is_gpu_inference, is_training)
        
        model_outputs.update(bbox_outputs)
        if not params['include_mask']:
            return model_outputs
        if params['delay_masks']:
            with tf.xla.experimental.jit_scope(compile_ops=False):
                cropped_gt_masks = preprocess_ops.preprocess_masks(labels['instance_masks'][0], labels['orig_boxes'][0], 
                                             features['image_info'][0], params)
        else:
            cropped_gt_masks = labels['cropped_gt_masks']
        mask_outputs = self.mask_roi_head(fpn_feats, rpn_box_rois, class_targets, box_targets, 
                                              proposal_to_label_map, params, features['image_info'], 
                                              model_outputs, is_gpu_inference, cropped_gt_masks,
                                              is_training)
        model_outputs.update(mask_outputs)
        return model_outputs 
    
    def feature_generator(self, images, params, is_training=True):
        backbone_feats = self.backbone(
            images,
            training=is_training,
        )
        fpn_feats = self.fpn(backbone_feats, training=is_training)
        rpn_score_outputs, rpn_box_outputs = self.rpn_head_fn(
                                                    features=fpn_feats,
                                                    min_level=params['min_level'],
                                                    max_level=params['max_level'],
                                                    is_training=is_training)
        return rpn_score_outputs, rpn_box_outputs, fpn_feats
    
    def roi_generator(self, rpn_score_outputs, rpn_box_outputs, params, 
                      image_height, image_width, image_info, labels, 
                      is_training=True):
        
        all_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                      params['num_scales'], params['aspect_ratios'],
                                      params['anchor_scale'],
                                      (image_height, image_width))
        if is_training:
            rpn_pre_nms_topn = params['train_rpn_pre_nms_topn']
            rpn_post_nms_topn = params['train_rpn_post_nms_topn']
            rpn_nms_threshold = params['train_rpn_nms_threshold']
        else:
            rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
            rpn_post_nms_topn = params['test_rpn_post_nms_topn']
            rpn_nms_threshold = params['test_rpn_nms_thresh']
        if params['use_custom_box_proposals_op']:
            rpn_box_scores, rpn_box_rois = roi_ops.custom_multilevel_propose_rois(
                scores_outputs=rpn_score_outputs,
                box_outputs=rpn_box_outputs,
                all_anchors=all_anchors,
                image_info=image_info,
                rpn_pre_nms_topn=rpn_pre_nms_topn,
                rpn_post_nms_topn=rpn_post_nms_topn,
                rpn_nms_threshold=rpn_nms_threshold,
                rpn_min_size=params['rpn_min_size']
            )
        else:
            rpn_box_scores, rpn_box_rois = roi_ops.multilevel_propose_rois(
                scores_outputs=rpn_score_outputs,
                box_outputs=rpn_box_outputs,
                all_anchors=all_anchors,
                image_info=features['image_info'],
                rpn_pre_nms_topn=rpn_pre_nms_topn,
                rpn_post_nms_topn=rpn_post_nms_topn,
                rpn_nms_threshold=rpn_nms_threshold,
                rpn_min_size=params['rpn_min_size'],
                bbox_reg_weights=None,
                use_batched_nms=params['use_batched_nms']
            )
        rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)
        
        if is_training:
            rpn_box_rois = tf.stop_gradient(rpn_box_rois)
            rpn_box_scores = tf.stop_gradient(rpn_box_scores)  # TODO Jonathan: Unused => Shall keep ?

            # Sampling
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = \
            training_ops.proposal_label_op(
                rpn_box_rois,
                labels['gt_boxes'],
                labels['gt_classes'],
                batch_size_per_im=params['batch_size_per_im'],
                fg_fraction=params['fg_fraction'],
                fg_thresh=params['fg_thresh'],
                bg_thresh_hi=params['bg_thresh_hi'],
                bg_thresh_lo=params['bg_thresh_lo']
            )
        return box_targets, class_targets, rpn_box_rois, proposal_to_label_map
    
    def box_roi_head(self, fpn_feats, rpn_box_rois, box_targets, class_targets,
                     params, image_info, is_gpu_inference, is_training=True):
        # Performs multi-level RoIAlign.
        bbox_outputs = dict()
        box_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=rpn_box_rois,
            output_size=7,
            is_gpu_inference=is_gpu_inference
        )
        class_outputs, box_outputs, _ = self.box_head(inputs=box_roi_features)
        
        if not is_training:
            if params['use_batched_nms']:
                generate_detections_fn = postprocess_ops.generate_detections_gpu

            else:
                generate_detections_fn = postprocess_ops.generate_detections_tpu
            
            detections = generate_detections_fn(
                class_outputs=class_outputs,
                box_outputs=box_outputs,
                anchor_boxes=rpn_box_rois,
                image_info=image_info,
                pre_nms_num_detections=params['test_rpn_post_nms_topn'],
                post_nms_num_detections=params['test_detections_per_image'],
                nms_threshold=params['test_nms'],
                bbox_reg_weights=params['bbox_reg_weights']
            )
            bbox_outputs.update({
                'num_detections': detections[0],
                'detection_boxes': detections[1],
                'detection_classes': detections[2],
                'detection_scores': detections[3],
            })
            # testing outputs
            bbox_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
                                  'box_outputs': box_outputs,
                                  'anchor_boxes': rpn_box_rois})
        else:  # is training
            if params['box_loss_type'] != "giou":
                encoded_box_targets = training_ops.encode_box_targets(
                    boxes=rpn_box_rois,
                    gt_boxes=box_targets,
                    gt_labels=class_targets,
                    bbox_reg_weights=params['bbox_reg_weights']
                )

            bbox_outputs.update({
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
            })
            if params['box_loss_type'] != 'giou':
                bbox_outputs['box_targets'] = encoded_box_targets
        return bbox_outputs
    
    def mask_roi_head(self, fpn_feats, rpn_box_rois, class_targets, box_targets, 
                      proposal_to_label_map, params, image_info, 
                      model_outputs, is_gpu_inference, cropped_gt_masks,
                      is_training=True): 
        mask_outputs = dict()
        # Mask sampling
        if not is_training:
            selected_box_rois = model_outputs['detection_boxes']
            class_indices = model_outputs['detection_classes']
            class_indices = tf.cast(class_indices, dtype=tf.int32)
        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction'])
            )

            class_indices = tf.cast(selected_class_targets, dtype=tf.int32)
            
        mask_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            features=fpn_feats,
            boxes=selected_box_rois,
            output_size=14,
            is_gpu_inference=is_gpu_inference
        )
        mask_predictions = self.mask_head(inputs=mask_roi_features, class_indices=class_indices)
        
        if is_training:
            mask_targets = training_ops.get_mask_targets(
                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=cropped_gt_masks,
                output_size=params['mrcnn_resolution']
            )
            mask_outputs.update({
                'mask_outputs': mask_predictions,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })
        else:
            mask_outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_predictions),
            })

        return mask_outputs
    
    def rpn_head_fn(self, features, min_level=2, max_level=6, is_training=True):
        scores_outputs = dict()
        box_outputs = dict()
        for level in range(min_level, max_level + 1):
            scores_outputs[level], box_outputs[level] = self.rpn(features[level],
                                                                 training=is_training)
        return scores_outputs, box_outputs
    
class TapeModel(object):
    
    def __init__(self, params, train_input_fn=None, eval_input_fn=None, is_training=True):
        self.params = params


        self.forward = MRCNN(self.params.values(), is_training=is_training)
        self.model_dir = self.params.model_dir
        train_params = dict(self.params.values(), batch_size=self.params.train_batch_size)
        self.train_tdf = iter(train_input_fn(train_params)) \
                            if train_input_fn else None
        eval_params = dict(self.params.values(), batch_size=self.params.eval_batch_size)
        self.eval_tdf = iter(eval_input_fn(eval_params).repeat()) \
                            if eval_input_fn else None
        self.optimizer, self.schedule = self.get_optimizer()
        self.epoch_num = 0

    def load_weights(self):
        chkp = tf.compat.v1.train.NewCheckpointReader(self.params.checkpoint)
        weights = [chkp.get_tensor(i) for i in eager_mapping.resnet_vars]
        self.forward.layers[0].set_weights(weights)
        
    def get_optimizer(self):
        if self.params.lr_schedule=='piecewise':
            schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.params.learning_rate_steps,
                                                                            [self.params.init_learning_rate] + \
                                                                            self.params.learning_rate_levels)
        elif self.params.lr_schedule=='cosine':
            schedule = tf.keras.experimental.CosineDecay(self.params.init_learning_rate,
                                                         self.params.total_steps,
                                                         alpha=0.001)
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
            opt = optimizers.NovoGrad(learning_rate=schedule,
                                          beta_1=self.params.beta1,
                                          beta_2=self.params.beta2,
                                          weight_decay=self.params.l2_weight_decay,
                                          exclude_from_weight_decay=['bias', 'beta', 'batch_normalization'])
        else:
            raise NotImplementedError
        if self.params.amp:
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        return opt, schedule
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, features, labels, 
                   sync_weights=False, 
                   sync_opt=False):
        loss_dict = dict()
        with tf.GradientTape() as tape:
            model_outputs = self.forward(features, labels, self.params.values(), True)
            loss_dict['total_rpn_loss'], loss_dict['rpn_score_loss'], \
                loss_dict['rpn_box_loss'] = losses.rpn_loss(
                    score_outputs=model_outputs['rpn_score_outputs'],
                    box_outputs=model_outputs['rpn_box_outputs'],
                    labels=labels,
                    params=self.params.values()
                )
            loss_dict['total_fast_rcnn_loss'], loss_dict['fast_rcnn_class_loss'], \
                loss_dict['fast_rcnn_box_loss'] = losses.fast_rcnn_loss(
                    class_outputs=model_outputs['class_outputs'],
                    box_outputs=model_outputs['box_outputs'],
                    class_targets=model_outputs['class_targets'],
                    box_targets=model_outputs['box_targets'],
                    rpn_box_rois=model_outputs['box_rois'],
                    image_info=features['image_info'],
                    params=self.params.values()
                )
            if self.params.include_mask:
                loss_dict['mask_loss'] = losses.mask_rcnn_loss(
                    mask_outputs=model_outputs['mask_outputs'],
                    mask_targets=model_outputs['mask_targets'],
                    select_class_targets=model_outputs['selected_class_targets'],
                    params=self.params.values()
                )
            else:
                loss_dict['mask_loss'] = 0.
            if self.params.optimizer_type in ['LAMB', 'Novograd']: # decoupled weight decay
                loss_dict['l2_regularization_loss'] = tf.constant(0.0)
            else:
                loss_dict['l2_regularization_loss'] = self.params.l2_weight_decay * tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in self.forward.trainable_variables
                    if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
                ])
            loss_dict['total_loss'] = loss_dict['total_rpn_loss'] \
                + loss_dict['total_fast_rcnn_loss'] + loss_dict['mask_loss'] \
                + loss_dict['l2_regularization_loss']
            if self.params.amp:
                scaled_loss = self.optimizer.get_scaled_loss(loss_dict['total_loss'])

        if is_herring():
            if MPI_is_distributed(True):
                tape = herring.DistributedGradientTape(tape)
            if self.params.amp:
                scaled_gradients = tape.gradient(scaled_loss, self.forward.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss_dict['total_loss'], self.forward.trainable_variables)
            global_gradient_clip_ratio = self.params.global_gradient_clip_ratio
            if global_gradient_clip_ratio > 0.0:
                all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
                (clipped_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clip_ratio,
                                use_norm=tf.cond(all_are_finite, lambda: tf.linalg.global_norm(gradients), lambda: tf.constant(1.0)))
                gradients = clipped_grads
        
            grads_and_vars = []
            # Special treatment for biases (beta is named as bias in reference model)
            # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
            for grad, var in zip(gradients, self.forward.trainable_variables):
                if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                    grad = 2.0 * grad
                grads_and_vars.append((grad, var))

            # self.optimizer.apply_gradients(zip(gradients, self.forward.trainable_variables))
            self.optimizer.apply_gradients(grads_and_vars)
            if MPI_is_distributed(True) and sync_weights:
                if MPI_rank(True)==0:
                    logging.info("Broadcasting variables")
                herring.broadcast_variables(self.forward.variables, 0)
            if MPI_is_distributed(True) and sync_opt:
                if MPI_rank(True)==0:
                    logging.info("Broadcasting optimizer")
                herring.broadcast_variables(self.optimizer.variables(), 0)        
        else:
            if MPI_is_distributed():
                tape = hvd.DistributedGradientTape(tape, compression=hvd.compression.NoneCompressor)
            if self.params.amp:
                scaled_gradients = tape.gradient(scaled_loss, self.forward.trainable_variables)
                gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            else:
                gradients = tape.gradient(loss_dict['total_loss'], self.forward.trainable_variables)
            global_gradient_clip_ratio = self.params.global_gradient_clip_ratio
            if global_gradient_clip_ratio > 0.0:
                all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
                (clipped_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=global_gradient_clip_ratio,
                                use_norm=tf.cond(all_are_finite, lambda: tf.linalg.global_norm(gradients), lambda: tf.constant(1.0)))
                gradients = clipped_grads
        
            grads_and_vars = []
            # Special treatment for biases (beta is named as bias in reference model)
            # Reference: https://github.com/ddkang/Detectron/blob/80f3295308/lib/modeling/optimizer.py#L109
            for grad, var in zip(gradients, self.forward.trainable_variables):
                if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                    grad = 2.0 * grad
                grads_and_vars.append((grad, var))

            # self.optimizer.apply_gradients(zip(gradients, self.forward.trainable_variables))
            self.optimizer.apply_gradients(grads_and_vars)

            if MPI_is_distributed() and sync_weights:
                if MPI_rank()==0:
                    logging.info("Broadcasting variables")
                hvd.broadcast_variables(self.forward.variables, 0)
            if MPI_is_distributed() and sync_opt:
                if MPI_rank()==0:
                    logging.info("Broadcasting optimizer")
                hvd.broadcast_variables(self.optimizer.variables(), 0)
        return loss_dict
    
    def initialize_model(self):
        features, labels = next(self.train_tdf)
        model_outputs = self.forward(features, labels, self.params.values(), True)
        self.load_weights()
    
    def train_epoch(self, steps, broadcast=False):
        if MPI_rank(is_herring())==0:
            logging.info("Starting training loop")
            p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            loss_history = []
        else:
            p_bar = range(steps)

        timings=[]
        for i in p_bar:
            if broadcast and i==0:
                b_w, b_o = True, True
            elif i==0:
                b_w, b_o = False, True
            else:
                b_w, b_o = False, False
            
            tstart = time.perf_counter()
            features, labels = next(self.train_tdf)
            loss_dict = self.train_step(features, labels, b_w, b_o)

            delta_t = time.perf_counter() - tstart
            timings.append(delta_t)
            if MPI_rank(is_herring())==0:
                loss_history.append(loss_dict['total_loss'].numpy())
                step = self.optimizer.iterations
                learning_rate = self.schedule(step)
                p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(mean(loss_history[-50:]), 
                                                                          learning_rate))
            #if i%500 == 0:
            #    timings = np.asarray(timings, np.float)
            #    print(f"average step time={np.mean(timings)} +/- {np.std(timings)}")
            #    timings = []
        if MPI_rank(is_herring()) == 0:
            print("Saving checkpoint...")
            self.epoch_num+=1
            self.save_model()
            
    def get_latest_checkpoint(self):
        try:
            return sorted([_ for _ in os.listdir(self.model_dir) if _.endswith(".h5")])[-1]
        except:
            return None

    def save_model(self):
        filename = os.path.join(self.model_dir, f'weights_{self.epoch_num:02d}.h5')
        f = h5py.File(filename,'w')
        weights = self.forward.get_weights()
        for i in range(len(weights)):
            f.create_dataset('weight'+str(i),data=weights[i])
        f.close()

    def load_model(self, filename):
        file=h5py.File(filename,'r')
        weights = []
        for i in range(len(file.keys())):
            weights.append(file['weight'+str(i)][:])
        self.forward.set_weights(weights)
    

    @tf.function            
    def predict(self, features):
        labels = None
        model_outputs = self.forward(features, labels, self.params.values(), False)
        model_outputs.update({
                'source_id': features['source_ids'],
                'image_info': features['image_info'],
            })
        return model_outputs
            
    def run_eval(self, steps, async_eval=False, use_ext=False):
        if MPI_rank(is_herring())==0:
            logging.info("Starting eval loop")
            p_bar = tqdm(range(steps), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        else:
            p_bar = range(steps)
        worker_predictions = dict()
        for i in p_bar:
            features = next(self.eval_tdf)['features']
            out = self.predict(features)
            out = evaluation.process_prediction_for_eval(out)
            for k, v in out.items():
                if k not in worker_predictions:
                    worker_predictions[k] = [v]
                else:
                    worker_predictions[k].append(v)
        coco = coco_metric.MaskCOCO()
        _preds = copy.deepcopy(worker_predictions)
        for k, v in _preds.items():
            _preds[k] = np.concatenate(v, axis=0)
        if MPI_rank(is_herring()) < 32:
            converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
            worker_source_ids = _preds['source_id']
        else:
            converted_predictions = []
            worker_source_ids = []
        MPI.COMM_WORLD.barrier()
        predictions_list = evaluation.gather_result_from_all_processes(converted_predictions)
        source_ids_list = evaluation.gather_result_from_all_processes(worker_source_ids)
        validation_json_file=self.params.val_json_file
        if MPI_rank(is_herring()) == 0:
            all_predictions = []
            source_ids = []
            for i, p in enumerate(predictions_list):
                if i < 32:
                    all_predictions.extend(p)
            for i, s in enumerate(source_ids_list):
                if i < 32:
                    source_ids.extend(s)
            if use_ext:
                args = [all_predictions, validation_json_file]
                if async_eval:
                    eval_thread = threading.Thread(target=evaluation.fast_eval,
                                                   name="eval-thread", args=args)
                    eval_thread.start()
                else:
                    evaluation.fast_eval(*args)
            else:
                args = [all_predictions, source_ids, True, validation_json_file]
                if async_eval:
                    eval_thread = threading.Thread(target=evaluation.compute_coco_eval_metric_n, 
                                                   name="eval-thread", args=args)
                    eval_thread.start()
                else:
                    evaluation.compute_coco_eval_metric_n(*args)