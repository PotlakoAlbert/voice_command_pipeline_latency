#!/bin/bash
#SBATCH -p batch
#SBATCH --exclude=mscluster129
#SBATCH -J mt-smoke
#SBATCH -o logs/mt-smoke-%j.out
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=4
cd ~/voice_command_pipeline_latency
PY=~/conda_envs/kws-cu118/bin/python
$PY src/mini_trail_check.py && $PY src/mini_train.py --arch dilated_resnet --b 0.1 --no-trail --epochs 2 --n-train 640 --n-val 128 --n-pos 100 --n-neg 200 --out results/mini_trail_smoke --ckpt-dir checkpoints/mini_trail_smoke
