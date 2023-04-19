#!/bin/bash
#SBATCH --nodes=1
# #SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --job-name=diff_train
#SBATCH --output=out/log.diff
#SBATCH -p burst 
#SBATCH -A ccsd


source ~/.bashrc
cd /lustre/or-scratch/cades-ccsd/z6f/generative-discrete-state-diffusion-models
conda activate /lustre/or-scratch/cades-ccsd/z6f/conda_envs/jump

module load gcc
module load cuda/11.2

srun -n 2 python linear_nois_train.py

# mpiexec -n 1 python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
