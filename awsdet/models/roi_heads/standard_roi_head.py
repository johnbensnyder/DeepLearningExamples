import tensorflow as tf
from awsdet import core, training
from .base_roi_head import BaseRoIHead
from ..builder import HEADS, build_head, build_roi_extractor
from awsdet.core import training_ops

@HEADS.register_module()
class StandardRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
       
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)
        
    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)
    
    def call(self,
             fpn_feats,
             img_info,
             proposals,
             gt_bboxes=None,
             gt_labels=None,
             gt_masks=None,
             training=True):
        model_outputs=dict()
        if training:
            box_targets, class_targets, rpn_box_rois, proposal_to_label_map = self.bbox_sampler(proposals,
                                                                                                gt_bboxes, 
                                                                                                gt_labels)
        else:
            rpn_box_rois = proposals
        
        box_roi_features = self.bbox_roi_extractor(fpn_feats, rpn_box_rois)
        
        class_outputs, box_outputs, _ = self.bbox_head(inputs=box_roi_features)
        
        if not training:
            model_outputs.update(self.detector(class_outputs, box_outputs, rpn_box_rois, img_info))
            model_outputs.update({'class_outputs': tf.nn.softmax(class_outputs),
                                  'box_outputs': box_outputs,
                                  'anchor_boxes': rpn_box_rois})
        else:
            if self.train_cfg.box_loss_type!="giou":
                encoded_box_targets = self.box_encoder(boxes=rpn_box_rois,
                                                       gt_bboxes=box_targets,
                                                       gt_labels=class_targets)
            model_outputs.update({
                'class_outputs': class_outputs,
                'box_outputs': box_outputs,
                'class_targets': class_targets,
                'box_targets': encoded_box_targets if self.train_cfg.box_loss_type!="giou" else box_targets,
                'box_rois': rpn_box_rois,
            })
            total_loss, class_loss, box_loss = self.bbox_head.loss(model_outputs['class_outputs'],
                                                                   model_outputs['box_outputs'],
                                                                   model_outputs['class_targets'],
                                                                   model_outputs['box_targets'],
                                                                   model_outputs['box_rois'],
                                                                   img_info)
            model_outputs.update({
                'total_loss_bbox': total_loss,
                'class_loss': class_loss,
                'box_loss': box_loss
            })
        if not self.with_mask:
            return model_outputs
        
        if not training:
            selected_box_rois = model_outputs['detection_boxes']
            class_indices = model_outputs['detection_classes']
            #class_indices = tf.cast(class_indices, dtype=tf.int32)
        
        else:
            selected_class_targets, selected_box_targets, \
            selected_box_rois, proposal_to_label_map = training_ops.select_fg_for_masks(
                class_targets=class_targets,
                box_targets=box_targets,
                boxes=rpn_box_rois,
                proposal_to_label_map=proposal_to_label_map,
                max_num_fg=int(self.train_cfg.batch_size_per_im * self.train_cfg.fg_fraction)
            )

            #class_indices = tf.cast(selected_class_targets, dtype=tf.int32)
            class_indices = selected_class_targets
            
        mask_roi_features = self.mask_roi_extractor(
                fpn_feats,
                selected_box_rois,
            )
        
        mask_outputs = self.mask_head(inputs=mask_roi_features, class_indices=class_indices)
        
        if training:
            mask_targets = training_ops.get_mask_targets(
                fg_boxes=selected_box_rois,
                fg_proposal_to_label_map=proposal_to_label_map,
                fg_box_targets=selected_box_targets,
                mask_gt_labels=gt_masks,
                output_size=self.train_cfg.mrcnn_resolution
            )

            model_outputs.update({
                'mask_outputs': mask_outputs,
                'mask_targets': mask_targets,
                'selected_class_targets': selected_class_targets,
            })
            
            mask_loss = self.mask_head.loss(model_outputs['mask_outputs'],
                                             model_outputs['mask_targets'],
                                             model_outputs['selected_class_targets'],)
            model_outputs.update({'mask_loss': mask_loss})

        else:
            model_outputs.update({
                'detection_masks': tf.nn.sigmoid(mask_outputs),
            })

        return model_outputs
    