import sys
from awsdet import models
from awsdet import datasets
from awsdet import core
from awsdet import training
from awsdet.utils.runner import Runner
from awsdet.training.schedulers import WarmupScheduler
from awsdet.datasets.coco import evaluation
from configs.mrcnn_config import config
import tensorflow as tf
from tqdm import tqdm
from statistics import mean

import horovod.tensorflow as hvd
from mpi4py import MPI
hvd.init()

devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([devices[hvd.rank()]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
tf.config.optimizer.set_jit(True)

detector = models.TwoStageDetector(backbone=config.backbone_cfg,
                                   neck=config.fpn_cfg,
                                   rpn_head=config.rpn_head_cfg,
                                   roi_head=config.roi_head_cfg,
                                   train_cfg=config.train_config,
                                   test_cfg=config.test_config)

train_tdf = iter(datasets.build_dataset(config.train_data)().repeat())
val_tdf = iter(datasets.build_dataset(config.test_data)().repeat())

if hvd.rank()==0:
    result = detector(next(train_tdf)[0], training=False)
    chkp = tf.compat.v1.train.NewCheckpointReader(config.backbone_checkpoint)
    weights = [chkp.get_tensor(i) for i in ['/'.join(i.name.split('/')[-2:]).split(':')[0] for i in detector.layers[0].weights]]
    detector.layers[0].set_weights(weights)
    
global_batch_size = config.train_data['batch_size'] * hvd.size()
steps_per_epoch = config.train_config.images//global_batch_size
learning_rate = config.train_config.base_lr/8 * global_batch_size
    
schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([steps_per_epoch * 8, steps_per_epoch * 11],
                                                                [learning_rate, learning_rate/10, learning_rate/100])

schedule = WarmupScheduler(schedule, learning_rate/10, steps_per_epoch//8)

optimizer = tf.keras.optimizers.SGD(learning_rate=schedule,
                                    momentum=0.9)

if config.train_config.fp16:
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    
model_runner = Runner(model=detector, 
                      train_cfg=config.train_config, 
                      test_cfg=config.test_config, 
                      optimizer=optimizer)

for epoch in range(config.train_config.num_epochs):
    if hvd.rank()==0:
        print("Starting epoch {0}".format(epoch+1))
    pbar = tqdm(range(steps_per_epoch), file=sys.stdout, 
                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if hvd.rank()==0 else range(steps_per_epoch)
    loss_history = []
    for step in pbar:
        model_outputs = model_runner.train_step(next(train_tdf), 
                                                sync_weights=step==0, 
                                                sync_opt=step==0)
        loss_history.append(model_outputs['total_loss'].numpy())
        loss_rolling_mean = mean(loss_history[-50:])
        current_learning_rate = schedule(optimizer.iterations).numpy()
        if hvd.rank()==0:
            pbar.set_description("Loss {0:.4f}, LR: {1:.4f}".format(loss_rolling_mean, 
                                                                    current_learning_rate))
    p_bar = tqdm(range(5000//global_batch_size + 1),
                 bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if hvd.rank()==0 else range(5000//global_batch_size + 1)
    predictions = [model_runner.predict(next(val_tdf)) for i in p_bar]
    args = [predictions, model_runner.test_cfg.annotations]
    evaluation.evaluate_results(*args, pbar=True)
    MPI.COMM_WORLD.Barrier()
