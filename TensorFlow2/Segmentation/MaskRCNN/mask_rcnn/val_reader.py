import functools
import glob

import tensorflow as tf

from mask_rcnn.utils.distributed_utils import MPI_is_distributed
from mask_rcnn.utils.distributed_utils import MPI_rank_and_size
from mask_rcnn.utils.distributed_utils import MPI_rank
from mask_rcnn.utils.distributed_utils import MPI_size

from mask_rcnn.dataloader_utils import dataset_parser

class ValReader(object):
    """Input reader for dataset."""

    def __init__(
        self,
        file_pattern,
        num_examples=0,
        use_instance_mask=True,
        seed=None
    ):

        self._file_pattern = file_pattern
        self._use_instance_mask = use_instance_mask
        self._seed = seed
    
    def __call__(self, params):
        batch_size = params['batch_size'] if 'batch_size' in params else 1
        files = glob.glob(self._file_pattern)
        
        try:
            seed = params['seed'] if not MPI_is_distributed() else params['seed'] * MPI_rank()
        except (KeyError, TypeError):
            seed = None
            
        if MPI_is_distributed():
            n_gpus = MPI_size()

        elif input_context is not None:
            n_gpus = input_context.num_input_pipelines

        else:
            n_gpus = 1
        
        dataset = tf.data.TFRecordDataset(files)
        _shard_idx, _num_shards = MPI_rank_and_size()
        dataset = dataset.shard(_shard_idx, _num_shards)
        parser = lambda x: dataset_parser(x, 'infer', params, self._use_instance_mask, seed=seed)
        dataset = dataset.map(parser , num_parallel_call=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size,drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

