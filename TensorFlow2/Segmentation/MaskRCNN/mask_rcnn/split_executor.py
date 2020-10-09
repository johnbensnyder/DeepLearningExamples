import tensorflow as tf
import sys
import os
sys.path.append('..')
from statistics import mean
from mask_rcnn import dataloader
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn.tf2.split_mask_rcnn_model import MaskRCNN_Parallel, TapeModel
from mask_rcnn.tf2.utils import warmup_scheduler, eager_mapping
from mask_rcnn.utils.distributed_utils import MPI_is_distributed, MPI_local_rank, MPI_rank
from mask_rcnn.training import losses, learning_rates
from mask_rcnn.utils.logging_formatter import logging
from tqdm import tqdm

params = mask_rcnn_params.default_config()

params.training_file_pattern = '/home/ubuntu/data/nv_coco/train*'
params.use_fake_data = False
params.include_mask = True
params.disable_data_options = True
params.checkpoint = '/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603'
params.eval_batch_size = 1
params.loop_mode = 'tape'
params.data_slack = False
params.finetune_bn = False
params.l2_weight_decay = 1e-4
params.optimizer_type = 'SGD'
params.seed = 42
params.dist_eval = False
params.train_batch_size = 1
params.lr_schedule = "piecewise"
params.learning_rate_steps = [6000, 8000]
params.init_learning_rate = 0.005
params.learning_rate_levels = [0.1, 0.01]
params.warmup_learning_rate = 0.0001
params.warmup_steps = 1500
params.momentum = 0.9
params.amp = True
params.xla = True

devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(devices[0:2], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')

os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": params.amp})
tf.config.optimizer.set_jit(params.xla)
tf.config.experimental.set_synchronous_execution(False)

train_input_fn = dataloader.InputReader(
                file_pattern=params.training_file_pattern,
                mode=tf.estimator.ModeKeys.TRAIN,
                num_examples=None,
                use_fake_data=params.use_fake_data,
                use_instance_mask=params.include_mask,
                seed=params.seed,
                disable_options=params.disable_data_options
            )

model = TapeModel(params, logical_devices, train_input_fn=train_input_fn)

model.initialize_model()

model.train_epoch(1000, broadcast=True)
