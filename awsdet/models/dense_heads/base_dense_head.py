from abc import ABCMeta, abstractmethod

import tensorflow as tf

class BaseDenseHead(tf.keras.Model, metaclass=ABCMeta):
    """Base class for Dense Heads"""
    def __init__(self, **kwargs):
        super(BaseDenseHead, self).__init__(**kwargs)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute loss of the head"""
        pass
    