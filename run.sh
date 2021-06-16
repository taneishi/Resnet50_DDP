#!/bin/bash
#PBS -l nodes=4:gold6128:ppn=1
#PBS -N resnet_mpi
#PBS -j oe
#PBS -o output_${PBS_NUM_NODES}.log

cd ${PBS_O_WORKDIR}
mkdir -p tmp

#PYTHON=/opt/intel/oneapi/pytorch/1.7.0/bin/python 
PYTHON=${HOME}/anaconda3/envs/torch/bin/python

mpirun -machinefile ${PBS_NODEFILE} ${PYTHON} main.py \
    --epochs 5 \
    --tmpname ${PBS_O_WORKDIR}/tmp/${PBS_JOBID} \
    --env_size PMI_SIZE \
    --env_rank PMI_RANK
