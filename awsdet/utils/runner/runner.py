import os
import logging
from awsdet.utils.runner import LogBuffer
from awsdet.utils.dist_utils import MPI_rank_and_size, MPI_is_distributed, MPI_rank
import tensorflow as tf
import horovod.tensorflow as hvd

class Runner(object):
    
    def __init__(self,
                 model,
                 train_cfg=None,
                 test_cfg=None,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 amp_enabled=False):
        self.model = model
        self.optimizer = optimizer 
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if isinstance(work_dir, str):
            self.work_dir = os.path.abspath(work_dir)
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            self.work_dir = work_dir
        self._model_name = self.model.__class__.__name__
        # self.rank, self.size = MPI_rank_and_size()
        self._rank, self._size = MPI_rank_and_size()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0
        self._amp_enabled = amp_enabled
        
    @property
    def model_name(self):
        """
        Name of the model, usually the module class name.
        """
        return self._model_name


    @property
    def local_rank(self):
        """
        Local rank of current process
        """
        return self._local_rank


    @property
    def rank(self):
        """
        Global rank of current process. (distributed training)
        """
        return self._rank

    @property
    def world_size(self):
        """
        Number of processes participating in the job.
        (distributed training)
        """
        return self._world_size

    @property
    def local_size(self):
        """
        Number of processes running in the same node as this runner.
        (distributed training)
        """
        return self._local_size

    @property
    def hooks(self):
        """
        A list of registered hooks.
        """
        return self._hooks

    @property
    def epoch(self):
        """
        Current epoch.
        """
        return self._epoch

    @property
    def iter(self):
        """
        Current iteration
        """
        return self._iter

    @property
    def inner_iter(self):
        """
        Iteration in an epoch.
        """
        return self._inner_iter

    @property
    def max_epochs(self):
        """
        Maximum training epochs.
        """
        return self._max_epochs

    @property
    def max_iters(self):
        """
        Maximum training iterations.
        """
        return self._max_iters
        
    def _add_file_handler(self,
                          logger,
                          filename=None,
                          mode='w',
                          level=logging.INFO):
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """
        Init the logger.
        Args:
            log_dir(str, optional): Log file directory. If not specified, no
                log file will be used.
            level (int or str): See the built-in python logging module.
        Returns:
            :obj:`~logging.Logger`: Python logger.
        """
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = '{}.log'.format(self.timestamp)
            log_file = os.path.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger
    
    def register_hook(self, hook, priority='NORMAL'):
        """
        Register a hook into the hook list.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)
    
    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
            
    @tf.function
    def train_step(self, data_batch, sync_weights=False, sync_opt=False):
        with tf.GradientTape() as tape:
            outputs = self.model(*data_batch, training=True)
            if self.train_cfg.fp16:
                scaled_loss = self.optimizer.get_scaled_loss(outputs['total_loss'])
        if MPI_is_distributed():
            tape = hvd.DistributedGradientTape(tape, compression=hvd.compression.NoneCompressor)
        if self.train_cfg.fp16:
            scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(outputs['total_loss'], self.forward.trainable_variables)
        if self.train_cfg.global_gradient_clip_ratio > 0.0:
            all_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
            (clipped_grads, _) = tf.clip_by_global_norm(gradients, 
                                                        clip_norm=self.train_cfg.global_gradient_clip_ratio,
                                                        use_norm=tf.cond(all_are_finite, 
                                                            lambda: tf.linalg.global_norm(gradients), 
                                                            lambda: tf.constant(1.0)))
            gradients = clipped_grads
        grads_and_vars = []
        for grad, var in zip(gradients, self.model.trainable_variables):
            if grad is not None and any([pattern in var.name for pattern in ["bias", "beta"]]):
                grad = 2.0 * grad
            grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)
        if MPI_is_distributed() and sync_weights:
            if MPI_rank()==0:
                logging.info("Broadcasting variables")
            hvd.broadcast_variables(self.model.variables, 0)
        if MPI_is_distributed() and sync_opt:
            if MPI_rank()==0:
                logging.info("Broadcasting optimizer")
            hvd.broadcast_variables(self.optimizer.variables(), 0)
        return outputs
    
    # TODO:
    # this only works with the dict output from val change data pipeline
    # to make training and val match
    @tf.function
    def predict(self, data_batch):
        model_outputs = self.model(data_batch['features'], data_batch.get('labels'), False)
        model_outputs.update({
                'source_id': data_batch['features']['source_ids'],
                'image_info': data_batch['features']['image_info'],
            })
        return model_outputs