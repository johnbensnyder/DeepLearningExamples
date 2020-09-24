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
worker_predictions = np.load("predictions_all.npy", allow_pickle=True).item()
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

print(f'Length of converted_predictions: {len(converted_predictions)}')
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
    args = [all_predictions, source_ids, True, validation_json_file]
    eval_thread = threading.Thread(target=compute_coco_eval_metric_n, name="eval-thread", args=args)
    eval_thread.start()
