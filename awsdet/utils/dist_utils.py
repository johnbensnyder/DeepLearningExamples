import os
import sys

__all__ = ["MPI_local_rank", "MPI_rank", "MPI_size", "MPI_rank_and_size", "MPI_is_distributed", "is_herring"]

import os
import sys

def is_herring():
    if "RUN_HERRING" in os.environ and os.environ["RUN_HERRING"] == '1':
        return True
    else:
        return False

if is_herring():
    import herring.tensorflow as herring

def MPI_is_distributed(run_herring=False):
    """Return a boolean whether a distributed training/inference runtime is being used.
    :return: bool
    """
    if run_herring:
        return herring.size() > 1

    else:
        if all([var in os.environ for var in ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]]):
            return True

        elif all([var in os.environ for var in ["SLURM_PROCID", "SLURM_NTASKS"]]):
            return True

        else:
            return False


def MPI_local_rank(run_herring=False):

    if run_herring:
        return herring.local_rank()
    else:
        if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

        elif "SLURM_LOCALID" in os.environ:
            return int(os.environ.get("SLURM_LOCALID"))

        else:
            return 0

def MPI_local_size(run_herring=False):
    if run_herring:
        return herring.local_size() # check this
    else:
        return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))

def MPI_rank(run_herring=False):
    if run_herring:
        return herring.rank()
    else:
        return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))


def MPI_size(run_herring=False):
    if run_herring:
        return herring.size()
    else:
        return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))


def MPI_rank_and_size(run_herring=False):
    if run_herring:
        return herring.rank(), herring.size()
    else:
        return MPI_rank(), MPI_size()


# Source: https://github.com/horovod/horovod/blob/c3626e/test/common.py#L25
def mpi_env_MPI_rank_and_size():
    """Get MPI rank and size from environment variables and return them as a
    tuple of integers.
    Most MPI implementations have an `mpirun` or `mpiexec` command that will
    run an MPI executable and set up all communication necessary between the
    different processors. As part of that set up, they will set environment
    variables that contain the rank and size of the MPI_COMM_WORLD
    communicator. We can read those environment variables from Python in order
    to ensure that `hvd.rank()` and `hvd.size()` return the expected values.
    Since MPI is just a standard, not an implementation, implementations
    typically choose their own environment variable names. This function tries
    to support several different implementation, but really it only needs to
    support whatever implementation we want to use for the TensorFlow test
    suite.
    If this is not running under MPI, then defaults of rank zero and size one
    are returned. (This is appropriate because when you call MPI_Init in an
    application not started with mpirun, it will create a new independent
    communicator with only one process in it.)

    Source: https://github.com/horovod/horovod/blob/c3626e/test/common.py#L25
    """
    rank_env = 'PMI_RANK SLURM_PROCID OMPI_COMM_WORLD_RANK'.split()
    size_env = 'PMI_SIZE SLURM_NTASKS OMPI_COMM_WORLD_SIZE'.split()

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)

    # Default to rank zero and size one if there are no environment variables
    return 0, 1

def get_dist_info(run_herring=False):
    rank = MPI_rank(run_herring)
    local_rank = MPI_local_rank(run_herring)
    size = MPI_size(run_herring)
    local_size = MPI_local_size(run_herring)
    return rank, local_rank, size, local_size

