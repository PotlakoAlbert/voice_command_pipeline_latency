#!/bin/bash
#SBATCH -p batch
#SBATCH --exclude=mscluster129
#SBATCH -J ms-smoke
#SBATCH -o logs/ms-smoke-%j.out
#SBATCH -t 00:40:00
#SBATCH --cpus-per-task=4
cd ~/voice_command_pipeline_latency
PY=~/conda_envs/kws-cu118/bin/python
$PY src/mini_shift_check.py && $PY src/mini_train_frame.py --arch dilated_resnet --b 0.5 --shift-ms 100 --epochs 2 --n-train 1280 --n-val 128 --n-pos 100 --n-neg 200 --out results/mini_shift_smoke --ckpt-dir checkpoints/mini_shift_smoke && $PY src/mini_train_frame.py --arch repcnn --b 0.5 --shift-ms 100 --epochs 1 --n-train 640 --n-val 128 --n-pos 100 --n-neg 200 --out results/mini_shift_smoke --ckpt-dir checkpoints/mini_shift_smoke
