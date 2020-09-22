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

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.disable_v2_behavior()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

import horovod.tensorflow as hvd
hvd.init()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[hvd.rank()], 'GPU')
devices = tf.config.list_logical_devices('GPU')

from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn import dataset_utils
import load_weights, model
from evaluation import compute_coco_eval_metric_nonestimator, process_prediction_for_eval, gather_result_from_all_processes

train_file_pattern = '/home/ubuntu/nv_tfrecords/train*'
batch_size = 6
data_params = dataset_params.get_data_params()
params = mask_rcnn_params.default_config().values()
data_params['batch_size'] = batch_size
params['finetune_bn'] = False
params['train_batch_size'] = batch_size
params['l2_weight_decay'] = 1e-4
params['init_learning_rate'] = 1e-4 * batch_size
params['warmup_learning_rate'] = 1e-3 * batch_size
params['warmup_steps'] = 500
params['learning_rate_steps'] = [30000,40000]
params['learning_rate_levels'] = [1e-4 * batch_size, 1e-5 * batch_size]
params['momentum'] = 0.9
params['use_batched_nms'] = True

data_params_eval = dataset_params.get_data_params()
data_params_eval['batch_size'] = 8

from mask_rcnn import dataloader
train_input_fn = dataloader.InputReader(
    file_pattern=train_file_pattern,
    mode=tf.estimator.ModeKeys.TRAIN,
    num_examples=None,
    use_fake_data=False,
    use_instance_mask=True,
)
train_tdf = train_input_fn(data_params)
tdf_iter = train_tdf.make_initializable_iterator()
features, labels = tdf_iter.get_next()


val_file_pattern = '/home/ubuntu/nv_tfrecords/val*'
val_input_fn = dataloader.InputReader(
    file_pattern=val_file_pattern,
    mode=tf.estimator.ModeKeys.PREDICT,
    num_examples=5000,
    use_fake_data=False,
    use_instance_mask=True,
)
val_tdf = val_input_fn(data_params_eval)
val_iter = val_tdf.make_initializable_iterator()
features_val = val_iter.get_next()

train_op, total_loss = model.model(features, params, labels, labels)
model_output = model.model(features_val['features'], params, is_training=False)

var_list = load_weights.build_assigment_map('resnet50/')
checkpoint_file = tf.train.latest_checkpoint('../resnet/resnet-nhwc-2018-02-07/')
_init_op, _init_feed_dict = load_weights.assign_from_checkpoint(checkpoint_file, var_list)

var_initializer = tf.global_variables_initializer()
loss_history = []
steps = 118000//(batch_size * hvd.size())
steps = 10
val_json_file="/home/ubuntu/nv_tfrecords/annotations/instances_val2017.json"



with tf.Session() as sess:
    sess.run(_init_op, _init_feed_dict)
    sess.run(tdf_iter.initializer)
    sess.run(val_iter.initializer)
    sess.run(var_initializer)
    for epoch in range(1):
        print("starting training")
        print(hvd.rank())
        if hvd.rank()==0:
            progressbar = tqdm(range(steps))
            loss_history = []
        else:
            progressbar = range(steps)
        for i in progressbar:
            op, loss = sess.run((train_op, total_loss))
            if hvd.rank()==0:
                print(loss)
                loss_history.append(loss)
                progressbar.set_description("Loss: {0:.4f}".format(np.array(loss_history[-50:]).mean()))

        eval_batch_size = 8
        eval_steps = 5000//(eval_batch_size * hvd.size())
        #eval_steps = 625
        progressbar_eval = tqdm(range(eval_steps))

        predictions = dict()
        for i in progressbar_eval:
            out= sess.run((model_output))
            out = process_prediction_for_eval(out)

            for k, v in out.items():
                if k not in predictions:
                    predictions[k] = [v]
                else:
                    predictions[k].append(v)
        
        predictions_all = gather_result_from_all_processes(predictions)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        comm.barrier()

        if(hvd.rank() == 0):
            print("#"*20)
            print(len(predictions_all))
        predictions_collected = dict()
        if(hvd.rank() == 0):
            for out in predictions_all:
                for k, v in out.items():
                    if k not in predictions_collected:
                        predictions_collected[k] = v
                    else:
                        predictions_collected[k] += v
        comm.barrier()
        
        #args = [predictions_collected, val_json_file]
        compute_coco_eval_metric_nonestimator(predictions_collected, val_json_file)
        #eval_thread.start()

#np.save("predictions_all.npy", predictions_all)
