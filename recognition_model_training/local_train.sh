#!/bin/bash
MODEL_ROOT=""
LOG_ROOT=""

if [ ! -d "${MODEL_ROOT}" ]; then
    echo ">> Creating checkpoint directory: ${MODEL_ROOT}"
    mkdir -p "${MODEL_ROOT}"
else:
    echo ">> Checkpoint directory exists: ${MODEL_ROOT}"
fi

if [ ! -d "${LOG_ROOT}" ]; then
    echo ">>Creating log directory: ${LOG_ROOT}"
    mkdir -p "${LOG_ROOT}"
else:
    echo ">> Log directory exists: ${LOG_ROOT}"
fi

# export CUDA_VISIBLE_DEVICES='0'
# python3 -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 tface_train.py

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
nohup python3 -u -m torch.distributed.launch  --nproc_per_node=8 --nnodes=1 tface_train.py > ${LOG_ROOT}/$(date +%F-%H-%M-%S).log 2>&1 &
