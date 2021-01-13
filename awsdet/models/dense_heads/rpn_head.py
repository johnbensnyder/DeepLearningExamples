import tensorflow as tf
from .anchor_head import AnchorHead
from awsdet import training
from awsdet import core
from ..builder import HEADS

@HEADS.register_module()
class RPNHead(AnchorHead):
    def __init__(self, **kwargs):
        super(RPNHead, self).__init__(1, **kwargs)
    
    def _init_layers(self):
        self.rpn_conv = tf.keras.layers.Conv2D(
                            self.feat_channels,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation=tf.nn.relu,
                            bias_initializer=tf.keras.initializers.Zeros(),
                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                            padding='same',
                            trainable=self.trainable,
                            name='rpn'
                        )
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
        
    def call(self, inputs, img_info, training=True, *args, **kwargs):
        scores_outputs = dict()
        box_outputs = dict()
        for level in range(self.anchor_generator.min_level, 
                           self.anchor_generator.max_level + 1):
            net = self.rpn_conv(inputs[level])
            scores_outputs[level] = self.conv_cls(net)
            box_outputs[level] = self.conv_reg(net)
        proposals = self.get_bboxes(scores_outputs,
                                    box_outputs,
                                    img_info,
                                    self.anchor_generator,
                                    training=training)
        return scores_outputs, box_outputs, proposals
        
