#!/usr/bin/bash -x
#SBATCH -p gpu_4
#SBATCH --job=CORE
#SBATCH --nodes=1
#SBATCH -c 4
#SBTACH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=output_file.out
#SBATCH --error=output_file.out
#SBATCH --mem=150gb
#SBATCH --gres gpu:1



cd $HOME/ml-core
source ~/.bashrc
conda activate core

export PYTHONPATH=.:$PYTHONPATH

MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python train.py --config configs/dmc/core.yaml
