from awsdet.utils import Registry, build_from_cfg

ANCHORS = Registry('Anchor generator')
ROIS = Registry('ROI')
DETECTIONS = Registry('Detection')
ENCODERS = Registry('Encoder')

def build_anchors(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHORS, default_args)

def build_roi(cfg, default_args=None):
    return build_from_cfg(cfg, ROIS, default_args)

def build_detections(cfg, default_args=None):
    return build_from_cfg(cfg, DETECTIONS, default_args)

def build_encoder(cfg, default_args=None):
    return build_from_cfg(cfg, ENCODERS, default_args)
