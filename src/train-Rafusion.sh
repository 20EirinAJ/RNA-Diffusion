#!/bin/bash

# Slurmジョブの設定
#SBATCH --job-name=my_gpu_job
#SBATCH --output=log/logfile.log  # ログファイルのパスを指定
#SBATCH --error=log/error.log    # エラーログファイルのパスを指定

export CUDA_PATH=/usr/local/cuda-11.8
export CUDA_HOME=${CUDA_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
python train.py           # ジョブの実行
sleep 100