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
sys.path.append('..')
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
sys.path.append('..')


os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ADJUST_HUE_FUSED'] = '1'
os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'
os.environ['HOROVOD_NUM_NCCL_STREAMS'] = '1'

import horovod.tensorflow as hvd
hvd.init()

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = str(hvd.size())
os.environ['TF_SYNC_ON_FINISH'] = '0'

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[hvd.rank()], True)
tf.config.set_visible_devices(physical_devices[hvd.rank()], 'GPU')
devices = tf.config.list_logical_devices('GPU')

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
from mask_rcnn import dataloader
from mask_rcnn import mask_rcnn_model
import load_weights, model_v2
from evaluation import compute_coco_eval_metric_nonestimator, process_prediction_for_eval, gather_result_from_all_processes

orig_file_pattern = '/home/ubuntu/nv_tfrecords/train*'
batch_size = 4
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
# params['finetune_bn'] = False
# params['train_batch_size'] = batch_size
# params['l2_weight_decay'] = 1e-4
# params['init_learning_rate'] = 2e-3 * batch_size * hvd.size()
# params['warmup_learning_rate'] = 2e-4 * batch_size * hvd.size()
# params['warmup_steps'] = 250
# params['learning_rate_steps'] = [30000,40000]
# params['learning_rate_levels'] = [2e-4 * batch_size * hvd.size(), 2e-5 * batch_size * hvd.size()]
# params['momentum'] = 0.9
# params['use_batched_nms'] = False
# params['use_custom_box_proposals_op'] = True
# params['amp'] = True
# params['include_groundtruth_in_features'] = True

params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 4e-2
params['warmup_learning_rate'] = 5e-3
params['warmup_steps'] = 250
params['learning_rate_steps'] = [30000,40000]
params['learning_rate_levels'] = [4e-3,
                                  4e-4]
params['momentum'] = 0.9
params['use_batched_nms'] = False
params['use_custom_box_proposals_op'] = False
params['amp'] = True
params['include_groundtruth_in_features'] = True
orig_loader = dataloader.InputReader(orig_file_pattern, use_instance_mask=True)
orig_tdf = orig_loader(data_params)
orig_iter = orig_tdf.make_initializable_iterator()
orig_features, orig_labels = orig_iter.get_next()

data_params_eval = dataset_params.get_data_params()
data_params_eval['batch_size'] = 4

val_file_pattern = '/home/ubuntu/nv_tfrecords/val*'
val_input_fn = dataloader.InputReader(
    file_pattern=val_file_pattern,
    use_instance_mask=True,
)
val_tdf = val_input_fn(data_params_eval)
val_iter = val_tdf.make_initializable_iterator()
features_val = val_iter.get_next()

mask_rcnn = model_v2.MRCNN(params)

train_outputs = model_v2.model_fn(orig_features, orig_labels, params, mask_rcnn, is_training=True)
model_output = model_v2.model_fn(features_val[0], None, params, mask_rcnn, is_training=False)

var_list = load_weights.build_assigment_map('mrcnn/resnet50/')
checkpoint_file = tf.train.latest_checkpoint('../resnet/resnet-nhwc-2018-02-07/')
_init_op, _init_feed_dict = load_weights.assign_from_checkpoint(checkpoint_file, var_list)

# sess = tf.Session()
# sess.run(orig_iter.initializer)
# sess.run(val_iter.initializer)
# sess.run(tf.global_variables_initializer())
# sess.run(_init_op, _init_feed_dict)
#saver = tf.train.Saver()
#saver.restore(sess,tf.train.latest_checkpoint('saved_model/'))

steps = 118000//(batch_size * hvd.size()) 
#steps = 10
loss_history = []
rpn_loss_history = []
rcnn_loss_history = []
val_json_file="/home/ubuntu/nv_tfrecords/annotations/instances_val2017.json"

with tf.Session() as sess:
    sess.run(orig_iter.initializer)
    sess.run(val_iter.initializer)
    sess.run(tf.global_variables_initializer())
    sess.run(hvd.broadcast_global_variables(0))
    sess.run(_init_op, _init_feed_dict)
    for epoch in range(1):
        if hvd.rank()==0:
            progressbar = tqdm(range(steps))
            loss_history = []
        else:
            progressbar = range(steps)
        for i in progressbar:
            outputs = sess.run((train_outputs))
            if hvd.rank()==0:
                loss_history.append(outputs[1])
                rpn_loss_history.append(outputs[2])
                rcnn_loss_history.append(outputs[4])
                smoothed_loss = mean(loss_history[-50:])
                smoothed_rpn_loss = mean(rpn_loss_history[-50:])
                smoothed_rcnn_loss = mean(rcnn_loss_history[-50:])
                progressbar.set_description("Loss: {0:.4f}".format(np.array(loss_history[-50:]).mean()))
    
        #if MPI_rank() == 0:
        eval_batch_size = 4
        eval_steps = 5000//(eval_batch_size * hvd.size())
        #eval_steps = 20
        progressbar_eval = tqdm(range(eval_steps))

        worker_predictions = dict()
        for i in progressbar_eval:
            try:
                out= sess.run((model_output))
                out = process_prediction_for_eval(out)

                for k, v in out.items():
                    if k not in worker_predictions:
                        worker_predictions[k] = [v]
                    else:
                        worker_predictions[k].append(v)
            except:
                break
                           
            #compute_coco_eval_metric_nonestimator(worker_predictions, annotation_json_file=val_json_file)            
        
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

        validation_json_file="/home/ubuntu/nv_tfrecords/annotations/instances_val2017.json"
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
            
            np.save("predictions_all_new_multi.npy", all_predictions)
            np.save("source_ids_new_multi.npy", source_ids)

