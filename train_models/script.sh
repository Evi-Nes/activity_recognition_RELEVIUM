#!/bin/bash
#SBATCH --gres=gpu:rtx3060:1 -t 0-20

cd /mnt/cephfs/home/nestoropo/train_models

source ~/.bashrc
source ~/venv/bin/activate 


echo "Training Models"
python accelerometer_models.py
