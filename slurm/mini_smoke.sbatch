#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-smoke
#SBATCH -o logs/mini-smoke-%j.out
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=4
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_train.py --arch dilated_resnet --b 0.1 --epochs 2 --n-train 640 --n-val 128 --n-pos 100 --n-neg 200 --out results/mini_smoke --ckpt-dir checkpoints/mini_smoke
