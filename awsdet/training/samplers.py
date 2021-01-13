from awsdet.core import training_ops
from .builder import SAMPLERS

@SAMPLERS.register_module()
class RandomSampler(object):
    
    def __init__(self, 
                 batch_size_per_im=512,
                 fg_fraction=0.25, 
                 fg_thresh=0.5,     
                 bg_thresh_hi=0.5, 
                 bg_thresh_lo=0.):
        self.batch_size_per_im=batch_size_per_im
        self.fg_fraction=fg_fraction
        self.fg_thresh=fg_thresh
        self.bg_thresh_hi=bg_thresh_hi
        self.bg_thresh_lo=bg_thresh_lo
        
    def __call__(self,
                 boxes, 
                 gt_boxes, 
                 gt_labels):
        sample_box_targets, class_targets, rois, sample_proposal_to_label_map = training_ops.proposal_label_op(boxes, 
                                               gt_boxes, 
                                               gt_labels,
                                               batch_size_per_im=self.batch_size_per_im,
                                               fg_fraction=self.fg_fraction,
                                               fg_thresh=self.fg_thresh,
                                               bg_thresh_hi=self.bg_thresh_hi,
                                               bg_thresh_lo=self.bg_thresh_lo,)
        return sample_box_targets, class_targets, rois, sample_proposal_to_label_map
        