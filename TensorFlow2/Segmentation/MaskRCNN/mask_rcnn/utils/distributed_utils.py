#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import herring.tensorflow as herring
__all__ = ["MPI_local_rank", "MPI_rank", "MPI_size", "MPI_rank_and_size", "MPI_is_distributed"]


def MPI_is_distributed():
    """Return a boolean whether a distributed training/inference runtime is being used.
    :return: bool
    """
    return herring.size() > 1


def MPI_local_rank():
    return herring.local_rank()


def MPI_rank():
    return herring.rank()


def MPI_size():
    return herring.size()


def MPI_rank_and_size():
    return herring.rank(), herring.size()

