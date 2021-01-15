from argparse import Namespace
import tensorflow as tf

class _Namespace(Namespace):
    def values(self):
        return self.__dict__

train_data = dict(
            type="CocoInputReader",
            file_pattern="/workspace/data/coco/train*",
            batch_size=2,
            mode=tf.estimator.ModeKeys.TRAIN,
            use_instance_mask=True,
            params=dict(
                visualize_images_summary=False,
                image_size=(832, 1344),
                min_level=2,
                max_level=6,
                num_scales=1,
                aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                anchor_scale=8.0,
                include_mask=True,
                skip_crowd_during_training=True,
                include_groundtruth_in_features=False,
                use_category=True,
                flatten_masks=False,
                augment_input_data=True,
                gt_mask_size=112,
                num_classes=91,
                rpn_positive_overlap=0.7,
                rpn_negative_overlap=0.3,
                rpn_batch_size_per_im=256,
                rpn_fg_fraction=0.5,
                )
            )

test_data = dict(
            type="CocoInputReader",
            file_pattern="/workspace/data/coco/val*",
            batch_size=2,
            mode=tf.estimator.ModeKeys.PREDICT,
            params=dict(
                visualize_images_summary=False,
                image_size=(832, 1344),
                min_level=2,
                max_level=6,
                num_scales=1,
                aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                anchor_scale=8.0,
                include_mask=True,
                skip_crowd_during_training=True,
                include_groundtruth_in_features=False,
                use_category=True,
                flatten_masks=True,
                augment_input_data=True,
                gt_mask_size=112,
                num_classes=91,
                rpn_positive_overlap=0.7,
                rpn_negative_overlap=0.3,
                rpn_batch_size_per_im=256,
                rpn_fg_fraction=0.5,
                ),
            dist_eval=True,
            )

train_config = _Namespace(**dict(
            base_lr=1e-2,
            num_epochs=13,
            fp16=True,
            xla=True,
            weight_decay=1e-4,
            batch_size_per_im=512,
            images=118287,
            fg_fraction=0.25,
            mrcnn_resolution=28,
            global_gradient_clip_ratio=0.0,
            box_loss_type="huber",
            sampler_cfg=dict(
                     type="RandomSampler",
                     ),
            rpn_loss_cfg = dict(
                    type="RPNLoss"
                    ),
            ))

test_config = _Namespace(**dict(
            annotations='/workspace/data/coco/annotations/instances_val2017.json',
            async_eval=False,
            ))

config = _Namespace(**dict(
            train_data=train_data,
            test_data=test_data,
            train_config=train_config,
            test_config=test_config,
            backbone_checkpoint="/model/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603",
            backbone_cfg=dict(
                    type='Resnet_Model',
                    resnet_model='resnet50',
                    data_format='channels_last', 
                    trainable=True, 
                    finetune_bn=False, 
                    norm_type='batchnorm'
                ),
            fpn_cfg=dict(
                    type="FPN",
                    min_level=2, 
                    max_level=6, 
                    filters=256, 
                    trainable=True
                ),
            rpn_head_cfg = dict(
                    type="RPNHead",
                ),
            roi_head_cfg = dict(
                    type="StandardRoIHead",
                    bbox_roi_extractor=dict(
                        type="GenericRoIExtractor",
                        output_size=7,
                        is_gpu_inference=True),
                    bbox_head = dict(
                            type="BBoxHead",
                            num_classes=91, 
                             mlp_head_dim=1024, 
                             name="box_head", 
                             trainable=True,
                             loss_cfg=dict(
                                     type="FastRCNNLoss",
                                     num_classes=91,
                                     box_loss_type='huber',
                                     bbox_reg_weights=(10., 10., 5., 5.),
                                     fast_rcnn_box_loss_weight=1.
                                 )),
                    mask_roi_extractor = dict(
                            type="GenericRoIExtractor",
                            output_size=14,
                            is_gpu_inference=True),
                    mask_head = dict(
                            type="MaskHead",
                            num_classes=91,
                            mrcnn_resolution=28,
                            is_gpu_inference=True,
                            name="mask_head",
                            trainable=True,
                            loss_cfg=dict(
                                     type="MaskRCNNLoss",
                                     mrcnn_weight_loss_mask=1.
                                 )),
                    detector_cfg=dict(
                            type="BoxDetector",
                            use_batched_nms=True,
                            rpn_post_nms_topn=1000,
                            detections_per_image=100,
                            test_nms=0.5,
                            bbox_reg_weights=(10., 10., 5., 5.),),
                    box_encoder_cfg=dict(
                            type="TargetEncoder",
                            bbox_reg_weights=(10., 10., 5., 5.)),
                    train_cfg = train_config,
        )))