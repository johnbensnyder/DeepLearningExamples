from abc import ABCMeta, abstractmethod
import tensorflow as tf

class BaseDetector(tf.keras.Model, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
    
    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None
    
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))
    
    @abstractmethod
    def extract_feats(self, imgs, training=True):
        """Extract features from images."""
        pass
    
    @abstractmethod
    def parse_losses(self, losses, weight_decay=0.):
        pass
    
    @abstractmethod
    def call(self, features, labels=None, training=True):
        pass
    