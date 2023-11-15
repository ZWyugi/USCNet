#!/bin/bash
#SBATCH -J kidney
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -A P00120210009
python -V


#python train.py --epochs 100 --input_size "224,224,128" --batch_size 1 --input_path /mntcephfs/lab_data/wangcm/fan/data/data --lr 0.01
python train.py --lr 0.0001 --weight_decay 0.00001 --task="[0]" --input_size "256, 256, 144" --batch_size 1 --epochs 50 --input_path /mntcephfs/lab_data/wangcm/fan/data/amos  --pretrain_seg /mntcephfs/lab_data/wangcm/fan/models/seg_net_params_epo-50.pkl --save_epoch 5
