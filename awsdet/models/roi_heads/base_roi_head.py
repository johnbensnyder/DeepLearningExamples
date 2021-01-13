from abc import ABCMeta, abstractmethod
import tensorflow as tf
from ..builder import build_shared_head, build_roi_extractor
from awsdet import core, training

class BaseRoIHead(tf.keras.Model, metaclass=ABCMeta):
    """Base class for RoIHeads."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 detector_cfg=None,
                 box_encoder_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(BaseRoIHead, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if detector_cfg is not None:
            self.detector = core.build_detections(detector_cfg)
        if shared_head is not None:
            self.shared_head = build_shared_head(shared_head)
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)
        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)
        if box_encoder_cfg is not None:
            self.box_encoder = core.build_encoder(box_encoder_cfg)
        if train_cfg is not None:
            self.bbox_sampler = training.build_sampler(
                self.train_cfg.sampler_cfg)
    
    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None
    
    @abstractmethod
    def init_bbox_head(self):
        """Initialize ``bbox_head``"""
        pass

    @abstractmethod
    def init_mask_head(self):
        """Initialize ``mask_head``"""
        pass
    
    @abstractmethod
    def call(self,
             x,
             img_info,
             proposal_list,
             gt_bboxes=None,
             gt_labels=None,
             gt_masks=None,
             training=True,
             **kwargs):
        pass
    