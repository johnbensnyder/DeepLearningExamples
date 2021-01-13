import tensorflow as tf
from awsdet.core import spatial_transform_ops
from awsdet.models.builder import ROI_EXTRACTORS

@ROI_EXTRACTORS.register_module()
class GenericRoIExtractor(object):
    
    def __init__(self, 
                 output_size, 
                 is_gpu_inference=True):
        self.output_size = output_size
        self.is_gpu_inference = is_gpu_inference
        
    def __call__(self, 
                 fpn_feats, 
                 rpn_box_rois):
        roi_features = spatial_transform_ops.multilevel_crop_and_resize(
            fpn_feats,
            rpn_box_rois,
            output_size=self.output_size,
            is_gpu_inference=self.is_gpu_inference
        )
        return roi_features
