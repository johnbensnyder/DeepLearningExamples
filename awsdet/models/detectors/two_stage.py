import tensorflow as tf

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.
    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)
        
        if neck is not None:
            self.neck = build_neck(neck)
        
        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)
        
        if roi_head is not None:
            self.roi_head = build_head(roi_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None
    
    def extract_feats(self, imgs, training=True):
        x = self.backbone(imgs, training=training)
        if self.with_neck:
            x = self.neck(x, training=training)
        return x
    
    def call(self, features, labels=None, training=True):
        feature_maps = self.extract_feats(features['images'])
        scores_outputs, box_outputs, proposals = self.rpn_head(feature_maps, 
                                                               features['image_info'], 
                                                               training=training)
        if training:
            model_outputs = self.roi_head(feature_maps, features['image_info'], proposals[0], 
                     gt_bboxes=labels['gt_boxes'], gt_labels=labels['gt_classes'],
                     gt_masks=labels['cropped_gt_masks'], training=training)
            total_rpn_loss, rpn_score_loss, rpn_box_loss = self.rpn_head.loss(scores_outputs, box_outputs, labels)
            model_outputs.update({"total_rpn_loss": total_rpn_loss,
                                  "rpn_score_loss": rpn_score_loss,
                                  "rpn_box_loss": rpn_box_loss})
            loss_dict = self.parse_losses(model_outputs)
            model_outputs['total_loss'] = loss_dict['total_loss']
            model_outputs['l2_loss'] = loss_dict['l2_loss']
        else:
            model_outputs = self.roi_head(feature_maps, features['image_info'], proposals[0], training=training)
        return model_outputs
    
    def parse_losses(self, losses):
        loss_dict = dict()
        loss_dict['bbox_loss'] = losses['total_loss_bbox']
        loss_dict['mask_loss'] = losses['mask_loss']
        loss_dict['rpn_loss'] = losses['total_rpn_loss']
        loss_dict['l2_loss'] = self.train_cfg.weight_decay * tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in self.trainable_variables
                    if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
                ])
        loss_dict['total_loss'] = losses['total_loss_bbox'] + losses['mask_loss'] \
                                   + losses['total_rpn_loss'] + loss_dict['l2_loss']
        return loss_dict
    
    @tf.function
    def train_step(self, features, labels, optimizer, fp16=False):
        with tf.GradientTape() as tape:
            model_outputs = self(features, labels)
            loss_dict = self.parse_losses(model_outputs)
            if fp16:
                scaled_loss = optimizer.get_scaled_loss(loss_dict['total_loss'])
        if fp16:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(loss_dict['total_loss'], self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_dict
        
        