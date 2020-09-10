#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import sys
import datetime

os.environ['TF_GPU_THREAD_MODE']="gpu_private"
os.environ['TF_GPU_THREAD_COUNT']="2"
os.environ['TF_CPP_VMODULE']="auto_mixed_precision=3,xla_compilation_cache=1"
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

sys.path.append('..')
from mask_rcnn import dataloader
from mask_rcnn.hyperparameters import mask_rcnn_params
from mask_rcnn.hyperparameters import dataset_params
from mask_rcnn import tf2_model
#from mask_rcnn import mask_only
from mask_rcnn.training import losses
print(f"TF version is {tf.__version__} located at {tf.__file__}")


# In[ ]:


os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible  --tf_xla_min_cluster_size=2"
#os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=fusible"
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


# In[ ]:


data_params = dataset_params.get_data_params()
data_params['image_size'] = [832, 1344]
model_params = mask_rcnn_params.default_config().values()
model_params['finetune_bn'] = False
model_params['use_batched_nms'] = True
model_params['train_batch_size'] = data_params['batch_size']
model_params['l2_weight_decay'] = 1e-4
model_params['include_mask'] = True
train_data = '/Datasets_local/coco/coco-2017/coco2017-TFRecords/train*'


# In[ ]:


data = dataloader.InputReader(train_data, use_instance_mask=True)
train_tdf = data(data_params)


# In[ ]:


model = tf2_model.MaskRCNN(model_params)
#model = mask_only.MaskRCNN(model_params)


# In[ ]:


train_gen = iter(train_tdf.prefetch(256))


# In[ ]:


features, labels = next(train_gen)


# In[ ]:


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')


# In[ ]:


@tf.function
def train_step(features, labels, params):
    with tf.GradientTape() as tape:
        model_outputs = model(features, labels)
        # rpn loss
        total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
            score_outputs=model_outputs['rpn_score_outputs'],
            box_outputs=model_outputs['rpn_box_outputs'],
            labels=labels,
            params=model_params
        )
        # frcnn loss
        total_fast_rcnn_loss, fast_rcnn_class_loss, fast_rcnn_box_loss = losses.fast_rcnn_loss(
            class_outputs=model_outputs['class_outputs'],
            box_outputs=model_outputs['box_outputs'],
            class_targets=model_outputs['class_targets'],
            box_targets=model_outputs['box_targets'],
            params=model_params
        )
        '''total_fast_rcnn_loss = 0
        fast_rcnn_class_loss = 0
        fast_rcnn_box_loss = 0'''
        # mask loss
        if params['include_mask']:
            mask_loss = losses.mask_rcnn_loss(
                mask_outputs=model_outputs['mask_outputs'],
                mask_targets=model_outputs['mask_targets'],
                select_class_targets=model_outputs['selected_class_targets'],
                params=model_params
            )
        else:
            mask_loss = 0
        # l2 decay
        trainable_variables = model.trainable_variables
        l2_regularization_loss = model_params['l2_weight_decay'] * tf.add_n([
            tf.nn.l2_loss(v)
            for v in trainable_variables
            if not any([pattern in v.name for pattern in ["batch_normalization", "bias", "beta"]])
        ])
        total_loss = total_rpn_loss + total_fast_rcnn_loss + mask_loss + l2_regularization_loss
        scaled_loss = optimizer.get_scaled_loss(total_loss)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss


# In[ ]:


train_step(features, labels, model_params)


# In[ ]:


features = {}
labels = {}
for i in tqdm(range(64)):
   features[i], labels[i] = next(train_gen)


# In[ ]:


progressbar = tqdm(range(64))
loss_history = []
for i in progressbar:
    #features, labels = next(train_gen)
    loss = train_step(features[i], labels[i], model_params)
    loss_history.append(loss.numpy())
    progressbar.set_description("Loss: {0:.4f}".format(np.array(loss_history[-50:]).mean()))


# In[ ]:


progressbar = tqdm(range(64))
loss_history = []
profile_base="/DLExamples/TensorFlow2/Segmentation/MaskRCNN/Profiles"
suffix=""
xlaflags=os.environ.get("TF_XLA_FLAGS","")
if "fusible" in xlaflags:
    suffix+="_fusible"
if "min_cluster_size" in xlaflags:
    pos=xlaflags.find("min_cluster_size=")
    if pos >=0:
        substr=xlaflags[pos+len("min_cluster_size=")]
        if substr.find(" ") == -1:
            suffix+="_csize"+substr
        else:
            suffix+="_csize"+substr[:substr.find(" ")]

if os.environ.get("TF_GPU_THREAD_MODE") == 'gpu_private':
    suffix+="_privThr_"+os.environ.get("TF_GPU_THREAD_COUNT","2")

def do_step_profile(profile_path,stepstr):
#    tf.profiler.experimental.start(profile_path)
    print(f"Saving profile to {profile_path}")
    for i in progressbar:
 #       with tf.profiler.experimental.Trace(f"{stepstr}_train",step_num=i,_r=1):
        feature, label = next(train_gen)
        loss = train_step(feature, label, model_params)
        loss_history.append(loss.numpy())
        progressbar.set_description("Loss: {0:.4f}".format(np.array(loss_history[-50:]).mean()))

if "TFLocal" in tf.__file__:
    profile_path=os.path.join(profile_base,f"LocalBuild_2.3plus{suffix}")
    stepstr="local"
    do_step_profile(profile_path,stepstr)
elif "TFUpstream" in tf.__file__:
    profile_path=os.path.join(profile_base,f"Upstream_2.3{suffix}")
    stepstr="upstream"
    do_step_profile(profile_path,stepstr)
else:
    profile_path=os.path.join(profile_base,f"nvidia_20.08{suffix}")
#    tf.profiler.experimental.start(profile_path)
    stepstr="nvidia"
    for i in progressbar:
        feature, label = next(train_gen)
        loss = train_step(feature, label, model_params)
        loss_history.append(loss.numpy())
        progressbar.set_description("Loss: {0:.4f}".format(np.array(loss_history[-50:]).mean()))
#tf.profiler.experimental.stop()


# In[ ]:


print(datetime.datetime.now())


# In[ ]:




