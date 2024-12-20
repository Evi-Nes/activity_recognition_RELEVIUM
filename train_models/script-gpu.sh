#!/bin/bash
#SBATCH -c4 --mem 16G --gres=gpu:rtx4080:1 -t 0-50

cd /mnt/cephfs/home/nestoropo/activity_recognition_RELEVIUM/train_models

source ~/.bashrc
source ~/venv_gpu/bin/activate 


echo "Training Models with gpu"
python accelerometer_models.py
