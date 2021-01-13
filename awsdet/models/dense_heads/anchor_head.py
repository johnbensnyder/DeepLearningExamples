import tensorflow as tf
from .base_dense_head import BaseDenseHead
from awsdet import training
from awsdet import models
from awsdet import core
from ..builder import HEADS

@HEADS.register_module()
class AnchorHead(BaseDenseHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605
    def __init__(self,
                 num_classes,
                 feat_channels=256,
                 trainable=True,
                 anchor_cfg=dict(
                    type="AnchorGenerator",
                    min_level=2, 
                    max_level=6, 
                    num_scales=1, 
                    aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)], 
                    anchor_scale=8.0, 
                    image_size=(832, 1344)
                    ),
                 roi_proposal_cfg=dict(
                     type="ProposeROIs",
                     ),
                 sampler_cfg=dict(
                     type="RandomSampler",
                     ),
                 rpn_loss_cfg = dict(
                    type="RPNLoss"
                    )
                ):
        super(AnchorHead, self).__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes
        self.loss_func = models.build_loss(rpn_loss_cfg)
        self.anchor_generator = core.build_anchors(anchor_cfg)
        self.roi_proposal = core.build_roi(roi_proposal_cfg)
        # self.sampler = training.build_sampler(sampler_cfg)
        self.trainable = trainable
        self._init_layers()
        
    def _init_layers(self):
        self.conv_cls = tf.keras.layers.Conv2D(
                            len(self.anchor_generator.aspect_ratios * \
                                self.anchor_generator.num_scales) * self.cls_out_channels,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                            padding='valid',
                            trainable=self.trainable,
                            name='rpn-class'
                        )
        self.conv_reg = tf.keras.layers.Conv2D(
                            len(self.anchor_generator.aspect_ratios * \
                                self.anchor_generator.num_scales) * 4,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                            padding='valid',
                            trainable=self.trainable,
                            name='rpn-box'
                        )
        
    def loss(self, cls_score, bbox_pred, labels):
        return self.loss_func(cls_score, bbox_pred, labels)
    
    def call(self, inputs, img_info, gt_boxes=None, gt_labels=None, training=True, *args, **kwargs):
        cls_scores = self.conv_cls(inputs)
        bbox_preds = self.conv_reg(inputs)
        proposals = self.get_bboxes(cls_scores,
                                    bbox_preds,
                                    img_info,
                                    self.anchor_generator,
                                    gt_boxes=gt_boxes,
                                    gt_labels=gt_labels,
                                    training=training)
        return cls_scores, bbox_preds, proposals
    
    def get_bboxes(self, 
                   cls_scores,
                   bbox_preds,
                   img_info,
                   anchors,
                   training=True):
        rpn_box_rois, rpn_box_scores = self.roi_proposal(cls_scores,
                                                           bbox_preds,
                                                           img_info,
                                                           anchors,
                                                           training=training)
        return rpn_box_rois, rpn_box_scores
    