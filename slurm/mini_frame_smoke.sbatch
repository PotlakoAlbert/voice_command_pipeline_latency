#!/bin/bash
#SBATCH -p batch
#SBATCH --exclude=mscluster129
#SBATCH -J mf-smoke
#SBATCH -o logs/mf-smoke-%j.out
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=4
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_train_frame.py --arch dilated_resnet --b 0.1 --epochs 2 --n-train 1280 --n-val 128 --n-pos 100 --n-neg 200 --out results/mini_frame_smoke --ckpt-dir checkpoints/mini_frame_smoke
~/conda_envs/kws-cu118/bin/python src/mini_train_frame.py --arch dilated_resnet --b 0.0 --pooled-ce --epochs 1 --n-train 1280 --n-val 128 --n-pos 100 --n-neg 200 --out results/mini_frame_smoke --ckpt-dir checkpoints/mini_frame_smoke
