# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NovoGrad for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typing import Union, Callable, Optional, List
from typeguard import typechecked
import re

class NovoGrad(tf.keras.optimizers.Optimizer):
    """Optimizer that implements NovoGrad.

    The NovoGrad Optimizer was first proposed in [Stochastic Gradient
    Methods with Layerwise Adaptive Moments for training of Deep
    Networks](https://arxiv.org/pdf/1905.11286.pdf) NovoGrad is a
    first-order SGD-based algorithm, which computes second moments per
    layer instead of per weight as in Adam. Compared to Adam, NovoGrad
    takes less memory, and has been found to be more numerically stable.
    (For more information on the computation please refer to this
    [link](https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html))

    Second order moment = exponential moving average of Layer-wise square
    of grads:
        v_t <-- beta_2 * v_{t-1} + (1-beta_2) * (g_t)^2
    First order moment in one of four modes:
        1. moment of grads normalized by v_t:
            m_t <- beta_1 * m_{t-1} + [ g_t / (sqrt(v_t)+epsilon)]
        2. moment similar to Adam: exponential moving average of grads
        normalized by v_t (set grad_averaging = True to use this):
            m_t <- beta_1 * m_{t-1} +
                   [(1 - beta_1) * (g_t / (sqrt(v_t) + epsilon))]
        3. weight decay adds a w_d term after grads are rescaled by
        1/sqrt(v_t) (set weight_decay > 0 to use this0:
            m_t <- beta_1 * m_{t-1} +
                   [(g_t / (sqrt(v_t) + epsilon)) + (w_d * w_{t-1})]
        4. weight decay + exponential moving average from Adam:
            m_t <- beta_1 * m_{t-1} +
                   [(1 - beta_1) * ((g_t / (sqrt(v_t + epsilon)) +
                   (w_d * w_{t-1}))]
    Weight update:
        w_t <- w_{t-1} - lr_t * m_t

    Example of usage:
    ```python
    opt = tfa.optimizers.NovoGrad(
        lr=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=0.001,
        grad_averaging=False
    )
    ```
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: FloatTensorLike = 0.0,
        exclude_from_weight_decay: Optional[List[str]] = None,
        grad_averaging: bool = False,
        amsgrad: bool = False,
        name: str = "NovoGrad",
        **kwargs
    ):
        r"""Construct a new NovoGrad optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            grad_averaging: determines whether to use Adam style exponential
                moving averaging for the first order moments.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        if weight_decay < 0.0:
            raise ValueError("Weight decay rate cannot be negative")
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("grad_averaging", grad_averaging)
        self.amsgrad = amsgrad
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay


    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var=var, slot_name="m", initializer="zeros")
        for var in var_list:
            self.add_slot(
                var=var, slot_name="v", initializer=tf.zeros(shape=[], dtype=var.dtype)
            )
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                one_minus_beta_2_t=1 - beta_2_t,
                one_minus_beta_1_t=1 - beta_1_t,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        weight_decay = self._get_hyper("weight_decay")
        grad_averaging = self._get_hyper("grad_averaging")

        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        v_t = tf.cond(
            tf.equal(self.iterations, 0),
            lambda: g_2,
            lambda: v * coefficients["beta_2_t"]
            + g_2 * coefficients["one_minus_beta_2_t"],
        )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon)
        
        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            grad += weight_decay * var

        grad = tf.cond(
                tf.logical_and(grad_averaging, tf.not_equal(self.iterations, 0)),
                lambda: grad * coefficients["one_minus_beta_1_t"],
                lambda: grad,
                )
        m = self.get_slot(var, "m")
        return tf.raw_ops.ResourceApplyKerasMomentum(
                var=var.handle,
                accum=m.handle,
                lr=coefficients["lr_t"],
                grad=grad,
                momentum=coefficients["beta_1_t"],
                use_locking=self._use_locking,
                use_nesterov=False,
                )

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
                (var_device, var_dtype)
                ) or self._fallback_apply_state(var_device, var_dtype)
        weight_decay = self._get_hyper("weight_decay")
        grad_averaging = self._get_hyper("grad_averaging")

        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        # v is just a scalar and does not need to involve sparse tensors.
        v_t = tf.cond(
            tf.equal(self.iterations, 0),
            lambda: g_2,
            lambda: v * coefficients["beta_2_t"]
            + g_2 * coefficients["one_minus_beta_2_t"],
        )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon)

        var_name = self._get_variable_name(var.name)

        if self._do_use_weight_decay(var_name):
            grad +=  weight_decay * tf.gather(var, indices)

        grad = tf.cond(
            tf.logical_and(grad_averaging, tf.not_equal(self.iterations, 0)),
            lambda: grad * coefficients["one_minus_beta_1_t"],
            lambda: grad,
        )
        m = self.get_slot(var, "m")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=m.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["beta_1_t"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "grad_averaging": self._serialize_hyperparameter("grad_averaging"),
            }
        )
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
            return param_name



# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc

import tensorflow as tf
from tensorflow_addons.utils import types

import warnings
from typeguard import typechecked
from typing import Optional


class AveragedOptimizerWrapper(tf.keras.optimizers.Optimizer, metaclass=abc.ABCMeta):
    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        sequential_update: Optional[bool] = None,
        name: str = "AverageOptimizer",
        **kwargs
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        if not isinstance(sequential_update, bool):
            raise TypeError("sequential_update must be of bool type")

        self._optimizer = optimizer

        if sequential_update is not None:
            warnings.warn(
                "The parameter `sequential_update` is redundant due to AutoGraph. "
                "This behavior is deprecated and in Addons 0.12, this will raise an error. ",
                DeprecationWarning,
            )

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average")

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True) :
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name)

    @abc.abstractmethod
    def average_op(self, var, average_var):
        raise NotImplementedError

    def _apply_average_op(self, train_op, var):
        average_var = self.get_slot(var, "average")
        return self.average_op(var, average_var)

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices
        )
        average_op = self._apply_average_op(train_op, var)
        return tf.group(train_op, average_op)

    def assign_average_vars(self, var_list):
        """Assign variables in var_list with their respective averages.

        Args:
            var_list: List of model variables to be assigned to their average.

        Returns:
            assign_op: The op corresponding to the assignment operation of
            variables to their average.

        Example:
        ```python
        model = tf.Sequential([...])
        opt = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(lr=2.0), 100, 10)
        model.compile(opt, ...)
        model.fit(x, y, ...)

        # Update the weights to their mean before saving
        opt.assign_average_vars(model.variables)

        model.save('model.h5')
        ```
        """
        assign_op = tf.group(
            [
                var.assign(self.get_slot(var, "average"), use_locking=self._use_locking)
                for var in var_list
                if var.trainable
            ]
        )
        return assign_op

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects,
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)


# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An implementation of the Stochastic Weight Averaging optimizer.

The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
et. al in the paper [Averaging Weights Leads to Wider Optima and Better
Generalization](https://arxiv.org/abs/1803.05407). The optimizer
implements averaging of multiple points along the trajectory of SGD.
This averaging has shown to improve model performance on validation/test
sets whilst possibly causing a small increase in loss on the training
set.
"""

class SWA(AveragedOptimizerWrapper):
    """This class extends optimizers with Stochastic Weight Averaging (SWA).

    The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
    et. al in the paper [Averaging Weights Leads to Wider Optima and
    Better Generalization](https://arxiv.org/abs/1803.05407). The optimizer
    implements averaging of multiple points along the trajectory of SGD. The
    optimizer expects an inner optimizer which will be used to apply the
    gradients to the variables and itself computes a running average of the
    variables every `k` steps (which generally corresponds to the end
    of a cycle when a cyclic learning rate is employed).

    We also allow the specification of the number of steps averaging
    should first happen after. Let's say, we want averaging to happen every `k`
    steps after the first `m` steps. After step `m` we'd take a snapshot of the
    variables and then average the weights appropriately at step `m + k`,
    `m + 2k` and so on. The assign_average_vars function can be called at the
    end of training to obtain the averaged_weights from the optimizer.

    Note: If your model has batch-normalization layers you would need to run
    the final weights through the data to compute the running mean and
    variance corresponding to the activations for each layer of the network.
    From the paper: If the DNN uses batch normalization we run one
    additional pass over the data, to compute the running mean and standard
    deviation of the activations for each layer of the network with SWA
    weights after the training is finished, since these statistics are not
    collected during training. For most deep learning libraries, such as
    PyTorch or Tensorflow, one can typically collect these statistics by
    making a forward pass over the data in training mode
    ([Averaging Weights Leads to Wider Optima and Better
    Generalization](https://arxiv.org/abs/1803.05407))

    Example of usage:

    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    ```
    """

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        start_averaging: int = 0,
        average_period: int = 10,
        name: str = "SWA",
        sequential_update: bool = True,
        **kwargs
    ):
        r"""Wrap optimizer with the Stochastic Weight Averaging mechanism.

        Args:
            optimizer: The original optimizer that will be used to compute and
                apply the gradients.
            start_averaging: An integer. Threshold to start averaging using
                SWA. Averaging only occurs at `start_averaging` iters, must
                be >= 0. If start_averaging = m, the first snapshot will be
                taken after the mth application of gradients (where the first
                iteration is iteration 0).
            average_period: An integer. The synchronization period of SWA. The
                averaging occurs every average_period steps. Averaging period
                needs to be >= 1.
            name: Optional name for the operations created when applying
                gradients. Defaults to 'SWA'.
            sequential_update: Bool. If False, will compute the moving average
                at the same time as the model is updated, potentially doing
                benign data races. If True, will update the moving average
                after gradient updates.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(optimizer, sequential_update, name, **kwargs)

        if average_period < 1:
            raise ValueError("average_period must be >= 1")
        if start_averaging < 0:
            raise ValueError("start_averaging must be >= 0")

        self._set_hyper("average_period", average_period)
        self._set_hyper("start_averaging", start_averaging)

    @tf.function(experimental_relax_shapes=True)
    def average_op(self, var, average_var):
        average_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, average_period),
        )
        # The average update should happen iff two conditions are met:
        # 1. A min number of iterations (start_averaging) have taken place.
        # 2. Iteration is one in which snapshot should be taken.
        checkpoint = start_averaging + num_snapshots * average_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            tf.print(self.iterations, start_averaging, average_period, num_snapshots)
            num_snapshots = tf.cast(num_snapshots, tf.float32)
            average_value = (average_var * num_snapshots + var) / (num_snapshots + 1.0)
            return average_var.assign(average_value, use_locking=self._use_locking)

        return average_var

    def get_config(self):
        config = {
            "average_period": self._serialize_hyperparameter("average_period"),
            "start_averaging": self._serialize_hyperparameter("start_averaging"),
        }
        base_config = super().get_config()
        return {**base_config, **config}

