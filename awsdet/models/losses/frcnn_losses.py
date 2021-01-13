from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion

import tensorflow as tf
from awsdet.core import box_utils
from ..builder import LOSSES
from .losses import _huber_loss, _giou_loss, _softmax_cross_entropy, _sigmoid_cross_entropy

DEBUG_LOSS_IMPLEMENTATION = False


if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
    from tensorflow.python.keras.utils import losses_utils
    ReductionV2 = losses_utils.ReductionV2
else:
    ReductionV2 = tf.keras.losses.Reduction

@LOSSES.register_module()
class FastRCNNLoss(object):

    def __init__(self, 
                 num_classes=91,
                 box_loss_type='huber',
                 bbox_reg_weights=(10., 10., 5., 5.),
                 fast_rcnn_box_loss_weight=1.):
        self.num_classes = num_classes
        self.box_loss_type = box_loss_type
        self.bbox_reg_weights = bbox_reg_weights
        self.fast_rcnn_box_loss_weight = fast_rcnn_box_loss_weight
    
    def _fast_rcnn_class_loss(self, 
                              class_outputs, 
                              class_targets_one_hot, 
                              normalizer=1.0):
        """Computes classification loss."""
        with tf.name_scope('fast_rcnn_class_loss'):
            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.

            class_loss = _softmax_cross_entropy(onehot_labels=class_targets_one_hot, logits=class_outputs)

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                class_loss /= normalizer

        return class_loss
    
    def _fast_rcnn_box_loss(self, 
                            box_outputs, 
                            box_targets, 
                            class_targets, 
                            loss_type='huber', 
                            normalizer=1.0, 
                            delta=1.):
        """Computes box regression loss."""
        # delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P2-P6 pyramid is about [0.1, 0.1, 0.2, 0.2].

        with tf.name_scope('fast_rcnn_box_loss'):
            mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2), [1, 1, 4])

            # The loss is normalized by the sum of non-zero weights before additional
            # normalizer provided by the function caller.
            if loss_type == 'huber':
                box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
            elif loss_type == 'giou':
                box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
            else:
                # box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs, weights=mask, delta=delta)
                raise NotImplementedError

            if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
                box_loss /= normalizer

        return box_loss
    
    def __call__(self, 
                 class_outputs, 
                 box_outputs, 
                 class_targets, 
                 box_targets, 
                 rpn_box_rois, 
                 image_info):
        with tf.name_scope('fast_rcnn_loss'):
            class_targets = tf.cast(class_targets, dtype=tf.int32)
            # Selects the box from `box_outputs` based on `class_targets`, with which
            # the box has the maximum overlap.
            batch_size, num_rois, _ = box_outputs.get_shape().as_list()
            box_outputs = tf.reshape(box_outputs, [batch_size, num_rois, self.num_classes, 4])

            box_indices = tf.reshape(
                class_targets +
                tf.tile(tf.expand_dims(tf.range(batch_size) * num_rois * \
                                       self.num_classes, 1), [1, num_rois]) +
                tf.tile(tf.expand_dims(tf.range(num_rois) * self.num_classes, 0), [batch_size, 1]),
                [-1]
            )
            box_outputs = tf.matmul(
                tf.one_hot(
                    box_indices,
                    batch_size * num_rois * self.num_classes,
                    dtype=box_outputs.dtype
                ),
                tf.reshape(box_outputs, [-1, 4])
            )
            if self.box_loss_type == 'giou':
                # decode outputs to move deltas back to coordinate space
                rpn_box_rois = tf.reshape(rpn_box_rois, [-1, 4])
                box_outputs = box_utils.decode_boxes(encoded_boxes=box_outputs, 
                                                     anchors=rpn_box_rois, 
                                                     weights=self.bbox_reg_weights)
                # Clip boxes FIXME: hardcoding for now
                box_outputs = box_utils.clip_boxes(box_outputs, 832., 1344.)
            
            box_outputs = tf.reshape(box_outputs, [batch_size, -1, 4])
            box_loss = self._fast_rcnn_box_loss(
                box_outputs=box_outputs,
                box_targets=box_targets,
                class_targets=class_targets,
                loss_type=self.box_loss_type,
                normalizer=1.0
            )
            
            box_loss *= self.fast_rcnn_box_loss_weight
            
            use_sparse_x_entropy = False
            
            _class_targets = class_targets \
                if use_sparse_x_entropy \
                else tf.one_hot(class_targets, self.num_classes)
            
            class_loss = self._fast_rcnn_class_loss(
                class_outputs=class_outputs,
                class_targets_one_hot=_class_targets,
                normalizer=1.0
            )
            
            total_loss = class_loss + box_loss
            
        return total_loss, class_loss, box_loss

@LOSSES.register_module()
class MaskRCNNLoss(object):
    def __init__(self, mrcnn_weight_loss_mask=1.):
        self.mrcnn_weight_loss_mask = mrcnn_weight_loss_mask
    
    def __call__(self, mask_outputs, mask_targets, select_class_targets):
        with tf.name_scope('mask_loss'):
            batch_size, num_masks, mask_height, mask_width = mask_outputs.get_shape().as_list()
            weights = tf.tile(
                tf.reshape(tf.greater(select_class_targets, 0), 
                           [batch_size, num_masks, 1, 1]),
                [1, 1, mask_height, mask_width]
            )
            weights = tf.cast(weights, tf.float32)
            
            loss = _sigmoid_cross_entropy(
                multi_class_labels=mask_targets,
                logits=mask_outputs,
                weights=weights,
                sum_by_non_zeros_weights=True
            )
            
            mrcnn_loss = self.mrcnn_weight_loss_mask * loss
            
        return mrcnn_loss