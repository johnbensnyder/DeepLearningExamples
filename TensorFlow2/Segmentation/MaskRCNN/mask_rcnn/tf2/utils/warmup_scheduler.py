import tensorflow as tf


class WarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear or exponential warmup
    """
    
    def __init__(self, schedule, initial_learning_rate, warmup_steps, warmup_type='linear', init_steps=0,
                 dtype=tf.float32):
        super(WarmupScheduler, self).__init__()
        self.schedule = schedule
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.warmup_type = warmup_type
        self.dtype = dtype
        self.schedule_learning_rate = self.schedule(0)
        self.init_steps = init_steps
        
    def compute_linear_warmup(self, step):
        if(step >= self.init_steps):
          return ((self.schedule_learning_rate*step) + (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
        return 0.0
    
    #TODO: Change to properly add init_steps 
    #TODO: Should init steps he part of warmup??
    @tf.function
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp>=(self.warmup_steps): #+ self.init_steps):
            return self.schedule(global_step_recomp)
        return self.compute_linear_warmup(global_step_recomp)
    
    def get_config(self):
        schedule_config = self.schedule.get_config()
        schedule_config['initial_learning_rate'] = self.initial_learning_rate
        schedule_config['warmup_steps'] = self.warmup_steps
        schedule_config['warmup_type'] = self.warmup_type



class SWAScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Wraps another learning rate scheduler to add a linear or exponential warmup
    """
    
    def __init__(self, main_schedule, averaging_schedule, initial_learning_rate,
                    warmup_steps, swa_steps, swa_averaging_steps, warmup_type='linear',
                    init_steps=0, dtype=tf.float32):
        super(SWAScheduler, self).__init__()
        self.main_schedule = main_schedule
        self.averaging_schedule = averaging_schedule
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.warmup_steps = tf.cast(warmup_steps, dtype)
        self.swa_steps = tf.cast(swa_steps, dtype)
        self.swa_averaging_steps = swa_averaging_steps
        self.warmup_type = warmup_type
        self.init_steps = init_steps
        self.dtype = dtype
        self.schedule_learning_rate = self.main_schedule(0)
 
    def compute_linear_warmup(self, step):
        if(step >= self.init_steps):
            return ((self.schedule_learning_rate*step) + (self.initial_learning_rate*(self.warmup_steps-step)))/self.warmup_steps
        return 0.0
 
    @tf.function
    def __call__(self, step):
        global_step_recomp = tf.cast(step, self.dtype)
        if global_step_recomp >= self.warmup_steps and global_step_recomp < self.swa_steps:
            return self.main_schedule(global_step_recomp)
        elif global_step_recomp >= self.swa_steps:
            return self.averaging_schedule((global_step_recomp - self.swa_steps) % self.swa_averaging_steps)
        else:
            return self.compute_linear_warmup(global_step_recomp)

    def get_config(self):
        schedule_config = self.schedule.get_config()
        schedule_config['initial_learning_rate'] = self.initial_learning_rate
        schedule_config['warmup_steps'] = self.warmup_steps
        schedule_config['swa_steps'] = self.swa_steps
        schedule_config['swa_averaging_steps'] = self.swa_averaging_steps
        schedule_config['warmup_type'] = self.warmup_type

