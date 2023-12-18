#!/bin/bash

#$ -l rt_G.small=1
#$ -j y
#$ -N Rafusion-sample
#$ -cwd
#$ -l h_rt=10:00:00

source ~/.bashrc
conda activate rafusion
module load cuda/12.1/12.1.1


python /home/acf16126sc/Rafusion/src/sample.py