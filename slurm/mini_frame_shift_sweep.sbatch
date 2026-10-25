#!/bin/bash
#SBATCH -p batch
#SBATCH --exclude=mscluster129
#SBATCH -J mfs-sweep
#SBATCH -o logs/mfs-sweep-%A_%a.out
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-53
cd ~/voice_command_pipeline_latency
ARCHS=(crnn dilated_resnet repcnn)
BS=(0.0 0.1 0.2 0.33 0.5 1.0)
SEED=$((SLURM_ARRAY_TASK_ID / 18))
R=$((SLURM_ARRAY_TASK_ID % 18))
ARCH=${ARCHS[$((R / 6))]}
B=${BS[$((R % 6))]}
~/conda_envs/kws-cu118/bin/python src/mini_train_frame.py --arch $ARCH --b $B --seed $SEED --epochs 30 --patience 5 --shift-ms 100 --out results/mini_frame_s100 --ckpt-dir checkpoints/mini_frame_s100
