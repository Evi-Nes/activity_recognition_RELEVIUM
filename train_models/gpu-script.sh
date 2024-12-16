#!/bin/bash
#SBATCH -c4 --mem 4G --gres=gpu:rtx3090ti:1 -t 0-20

cd /mnt/cephfs/home/nestoropo/activity_recognition_RELEVIUM/train_models

source ~/.bashrc
source ~/venv_gpu/bin/activate 


echo "Training Models with gpu"
python accelerometer_models.py
