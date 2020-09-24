from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import operator
import pprint
import six
import time
import itertools
import collections
import io
import threading
import sys
sys.path.append('/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN')
from PIL import Image
from evaluation import *
import numpy as np
import tensorflow as tf

from mask_rcnn.utils.logging_formatter import logging

from mask_rcnn import coco_metric
from mask_rcnn.utils import coco_utils

from mask_rcnn.object_detection import visualization_utils
from mask_rcnn.utils.distributed_utils import MPI_rank

import dllogger
from dllogger import Verbosity
import numpy as np


import os
import sys
import itertools
from statistics import mean
from time import time
from tqdm import tqdm
import numpy as np
import threading

import os
import sys
sys.path.append('..')
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[hvd.rank()], True)
tf.config.set_visible_devices(physical_devices[hvd.local_rank()], 'GPU')
devices = tf.config.list_logical_devices('GPU')

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

from tqdm import tqdm
from statistics import mean

from mask_rcnn.tf2_model import MaskRCNN
from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from mask_rcnn.training import losses, learning_rates
from simple_model.tf2 import weight_loader, train, scheduler
from simple_model import model_v2



def do_eval(worker_predictions):
    print(f'Length of worker_predictions: {len(worker_predictions)}')
    logging.info(worker_predictions['source_id'])
    # DEBUG - print worker predictions
    # _ = compute_coco_eval_metric_n(worker_predictions, False, validation_json_file)
    coco = coco_metric.MaskCOCO()
    _preds = copy.deepcopy(worker_predictions)
    for k, v in _preds.items():
        # combined all results in flat structure for eval
        _preds[k] = np.concatenate(v, axis=0)
    if MPI_rank() < 32:
        converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
        worker_source_ids = _preds['source_id']
    else:
        converted_predictions = []
        worker_source_ids = []
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm.barrier()
    rank = MPI_rank()
    filename = "worker_source_id_" + str(rank) + ".npy"
    np.save(filename, worker_source_ids)
    print(len(worker_source_ids), len(set(worker_source_ids)), MPI_rank())
    #print(f'Length of converted_predictions: {len(converted_predictions)}')
    # logging.info(converted_predictions)
    # gather on rank 0
    predictions_list = gather_result_from_all_processes(converted_predictions)
    source_ids_list = gather_result_from_all_processes(worker_source_ids)

    validation_json_file="/home/ubuntu/data/annotations/instances_val2017.json"
    if MPI_rank() == 0:
        all_predictions = []
        source_ids = []
        for i, p in enumerate(predictions_list):
            if i < 32: # max eval workers (TODO config)
                all_predictions.extend(p)
        for i, s in enumerate(source_ids_list):
            if i < 32:
                source_ids.extend(s)

        # run metric calculation on root node TODO: launch this in it's own thread
        #compute_coco_eval_metric_n(all_predictions, source_ids, True, validation_json_file)
        
        args = [all_predictions, source_ids, True, validation_json_file]
        eval_thread = threading.Thread(target=compute_coco_eval_metric_n, name="eval-thread", args=args)
        eval_thread.start()


train_file_pattern = '/home/ubuntu/data/coco/train*'
batch_size = 1
eval_batch_size = 4
images = 118287
global_batch_size = batch_size * hvd.size()
steps_per_epoch = images//(batch_size * hvd.size())
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 1e-2 * global_batch_size / 8
params['warmup_learning_rate'] = 1e-3 * global_batch_size / 8
params['warmup_steps'] = 2048//hvd.size()
params['learning_rate_steps'] = [steps_per_epoch * 9, steps_per_epoch * 11]
params['learning_rate_levels'] = [1e-3 * global_batch_size / 8, 1e-4 * global_batch_size / 8]
params['momentum'] = 0.9
params['use_batched_nms'] = False
params['use_custom_box_proposals_op'] = True
params['amp'] = True
params['include_groundtruth_in_features'] = True

loader = dataset_utils.FastDataLoader(train_file_pattern, data_params)
train_tdf = loader(data_params)
train_tdf = train_tdf.apply(tf.data.experimental.prefetch_to_device(devices[0].name, 
                                                                    buffer_size=tf.data.experimental.AUTOTUNE))
train_iter = iter(train_tdf)

data_params_eval = dataset_params.get_data_params()
data_params_eval['batch_size'] = 4

val_file_pattern = '/home/ubuntu/data/coco/val*'
val_loader = dataset_utils.FastDataLoader(val_file_pattern, data_params_eval)
val_tdf = val_loader(data_params_eval)
val_tdf = val_tdf.apply(tf.data.experimental.prefetch_to_device(devices[0].name,
                                                                    buffer_size=tf.data.experimental.AUTOTUNE))

val_iter = iter(val_tdf)



mask_rcnn = model_v2.MRCNN(params)

features, labels = next(train_iter)
#features_val, _ = next(val_iter)

model_outputs = mask_rcnn(features, labels, params, is_training=True)

weight_loader.load_resnet_checkpoint(mask_rcnn, '/home/ubuntu/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/resnet/resnet-nhwc-2018-02-07/')

schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(params['learning_rate_steps'],
                                                                [params['init_learning_rate']] \
                                                                + params['learning_rate_levels'])
schedule = scheduler.WarmupScheduler(schedule, params['warmup_learning_rate'],
                                     params['warmup_steps'])
optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')

@tf.function
def train_step(features, labels, params, model, opt, first=False):
    with tf.GradientTape() as tape:
        total_loss = train.train_forward(features, labels, params, model)
        scaled_loss = optimizer.get_scaled_loss(total_loss)
    tape = hvd.DistributedGradientTape(tape)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if first:
        hvd.broadcast_variables(model.variables, 0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return total_loss

@tf.function
def pred(features, params):
    out = mask_rcnn(features, None, params, is_training=False)
    out['image_info'] = features['image_info']
    out['source_id'] = features['source_ids']
    return out


_ = train_step(features, labels, params, mask_rcnn, optimizer, first=True)


for epoch in range(20):

    if hvd.rank()==0:
        print(f'Starting Epoch {epoch}')
        p_bar = tqdm(range(steps_per_epoch))
        loss_history = []
    else:
        p_bar = range(steps_per_epoch)
    for i in p_bar:
        features, labels = next(train_iter)
        total_loss = train_step(features, labels, params, mask_rcnn, optimizer)
        if hvd.rank()==0:
            loss_history.append(total_loss.numpy())
            smoothed_loss = mean(loss_history[-50:])
            p_bar.set_description("Loss: {0:.4f}, LR: {1:.4f}".format(smoothed_loss, 
                                                                      schedule(optimizer.iterations)))
    

    eval_steps = 5000//(eval_batch_size * hvd.size())
    progressbar_eval = tqdm(range(eval_steps))
    worker_predictions = dict()    
    if hvd.rank()==0:
        print("Beginning eval")
        progressbar_eval = tqdm(range(eval_steps))
    else:
        progressbar_eval = range(eval_steps)

    for i in progressbar_eval:
        features_val, _ = next(val_iter)
        out = pred(features_val, params)
        out = process_prediction_for_eval(out)

        for k, v in out.items():
            if k not in worker_predictions:
                worker_predictions[k] = [v]
            else:
                worker_predictions[k].append(v)

    logging.info(worker_predictions['source_id'])
    # DEBUG - print worker predictions
    # _ = compute_coco_eval_metric_n(worker_predictions, False, validation_json_file)
    coco = coco_metric.MaskCOCO()
    _preds = copy.deepcopy(worker_predictions)
    for k, v in _preds.items():
        # combined all results in flat structure for eval
        _preds[k] = np.concatenate(v, axis=0)
    if MPI_rank() < 32:
        converted_predictions = coco.load_predictions(_preds, include_mask=True, is_image_mask=False)
        worker_source_ids = _preds['source_id']
    else:
        converted_predictions = []
        worker_source_ids = []
    
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    comm.barrier()
    rank = MPI_rank()
    #print(f'Length of converted_predictions: {len(converted_predictions)}')
    # logging.info(converted_predictions)
    # gather on rank 0
    predictions_list = gather_result_from_all_processes(converted_predictions)
    source_ids_list = gather_result_from_all_processes(worker_source_ids)

    validation_json_file="/home/ubuntu/data/annotations/instances_val2017.json"
    if MPI_rank() == 0:
        all_predictions = []
        source_ids = []
        for i, p in enumerate(predictions_list):
            if i < 32: # max eval workers (TODO config)
                all_predictions.extend(p)
        for i, s in enumerate(source_ids_list):
            if i < 32:
                source_ids.extend(s)

        # run metric calculation on root node TODO: launch this in it's own thread
        #compute_coco_eval_metric_n(all_predictions, source_ids, True, validation_json_file)
        
        args = [all_predictions, source_ids, True, validation_json_file]
        eval_thread = threading.Thread(target=compute_coco_eval_metric_n, name="eval-thread", args=args)
        eval_thread.start()    
