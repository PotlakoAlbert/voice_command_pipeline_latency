#!/bin/bash
#SBATCH -p batch
#SBATCH -J ep-sweep
#SBATCH -o logs/ep-sweep-%A_%a.out
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd ~/voice_command_pipeline_latency
ARCHS=(crnn dilated_resnet repcnn)
TLS=(0.0 0.1 0.2 0.3)
ARCH=${ARCHS[$((SLURM_ARRAY_TASK_ID / 4))]}
TL=${TLS[$((SLURM_ARRAY_TASK_ID % 4))]}
~/conda_envs/kws-cu118/bin/python src/train_kws.py --arch $ARCH --loss endpoint --target-latency $TL --data data --checkpoint-dir checkpoints --results-dir results --epochs 30 --patience 5 --num-workers 4
