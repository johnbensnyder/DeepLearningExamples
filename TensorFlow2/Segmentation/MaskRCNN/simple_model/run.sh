#!/bin/bash

#source activate tensorflow2_latest_p37
mpirun --allow-run-as-root \
    -np 8 \
    python train_dl.py
#/opt/amazon/openmpi/bin/mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
#    -N 2 \
#    -x NCCL_DEBUG=VERSION \
#    -x LD_LIBRARY_PATH \
#    -x PATH \
#    --oversubscribe \
#    python train_dl.py
