#!/bin/bash
#PBS -l nodes=4:ppn=1
#PBS -N resnet_mpi
#PBS -j oe
#PBS -o log/output_${PBS_NUM_NODES}.log

cd ${PBS_O_WORKDIR}
mkdir -p tmp log

mpirun -machinefile ${PBS_NODEFILE} python main.py \
    --epochs 2 \
    --tmpname ${PBS_O_WORKDIR}/tmp/${PBS_JOBID} \
    --env_size PMI_SIZE \
    --env_rank PMI_RANK
