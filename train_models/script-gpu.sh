#!/bin/bash
#SBATCH -c4 --mem 12G --gres shard:rtx4090:18 -t 0-50

cd /mnt/cephfs/home/nestoropo/activity_recognition_RELEVIUM/train_models

source ~/.bashrc
source ~/venv_gpu/bin/activate 


echo "Training Models with gpu"
python accelerometer_models.py
