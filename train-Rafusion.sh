#!/bin/bash

#$ -l rt_F=1
#$ -j y
#$ -N Rafusion-train
#$ -cwd
#$ -l h_rt=10:00:00

source ~/.bashrc
conda activate rafusion
module load cuda/12.1/12.1.1


python /home/acf16126sc/Rafusion/src/train.py