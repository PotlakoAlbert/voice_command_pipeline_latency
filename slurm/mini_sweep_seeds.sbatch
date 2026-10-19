#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-sweep
#SBATCH --exclude=mscluster129
#SBATCH -o logs/mini-sweep-%A_%a.out
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-17
cd ~/voice_command_pipeline_latency
ARCHS=(crnn dilated_resnet repcnn)
BS=(0.0 0.1 0.2 0.33 0.5 1.0)
ARCH=${ARCHS[$((SLURM_ARRAY_TASK_ID / 6))]}
B=${BS[$((SLURM_ARRAY_TASK_ID % 6))]}
~/conda_envs/kws-cu118/bin/python src/mini_train.py --arch $ARCH --b $B --seed ${SEED} --epochs 30 --patience 5
