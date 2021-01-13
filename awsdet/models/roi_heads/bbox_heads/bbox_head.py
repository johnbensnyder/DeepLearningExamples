import tensorflow as tf

from awsdet.models.builder import HEADS, build_loss

__all__ = ["BBoxHead"]

@HEADS.register_module()
class BBoxHead(tf.keras.Model):
    def __init__(self, 
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
                     ),
                 *args, 
                 **kwargs):
        """Box and class branches for the Mask-RCNN model.

        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: a integer for the number of classes.
        mlp_head_dim: a integer that is the hidden dimension in the fully-connected
          layers.
        """
        super(BBoxHead, self).__init__(name=name, trainable=trainable, *args, **kwargs)
        
        self._num_classes = num_classes
        self._mlp_head_dim = mlp_head_dim
        
        self._bbox_dense_0 = tf.keras.layers.Dense(
                units=mlp_head_dim,
                activation=tf.nn.relu,
                trainable=trainable,
                name='bbox_dense_0'
            )
        
        self._bbox_dense_1 = tf.keras.layers.Dense(
                units=mlp_head_dim,
                activation=tf.nn.relu,
                trainable=trainable,
                name='bbox_dense_1'
            )
        
        self._dense_class = tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                bias_initializer=tf.keras.initializers.Zeros(),
                trainable=trainable,
                name='class-predict'
            )
        
        self._dense_box = tf.keras.layers.Dense(
                num_classes * 4,
                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                bias_initializer=tf.keras.initializers.Zeros(),
                trainable=trainable,
                name='box-predict'
            )
        
        self.loss = build_loss(loss_cfg)
        
        
    def call(self, inputs, **kwargs):
        """
        Returns:
        class_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes], representing the class predictions.
        box_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes * 4], representing the box predictions.
        box_features: a tensor with a shape of
          [batch_size, num_rois, mlp_head_dim], representing the box features.
        """

        # reshape inputs before FC.
        batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()

        net = tf.reshape(inputs, [batch_size, num_rois, height * width * filters])

        net = self._bbox_dense_0(net)

        box_features = self._bbox_dense_1(net)

        class_outputs = self._dense_class(box_features)

        box_outputs = self._dense_box(box_features)

        return class_outputs, box_outputs, box_features