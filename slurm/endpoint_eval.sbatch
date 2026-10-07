#!/bin/bash
#SBATCH -p batch
#SBATCH -J ep-eval
#SBATCH -o logs/ep-eval-%A_%a.out
#SBATCH -t 04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd ~/voice_command_pipeline_latency
ARCHS=(crnn dilated_resnet repcnn)
TLS=(0.0 0.1 0.2 0.3)
ARCH=${ARCHS[$((SLURM_ARRAY_TASK_ID / 4))]}
TL=${TLS[$((SLURM_ARRAY_TASK_ID % 4))]}
SUFFIX=best
if [ "$ARCH" = "repcnn" ]; then SUFFIX=best_fused; fi
~/conda_envs/kws-cu118/bin/python src/evaluate_kws_latency.py --arch $ARCH --checkpoint checkpoints/${ARCH}_ep_tl${TL}_${SUFFIX}.pt --b 1.0 --output results/${ARCH}_ep_tl${TL}_latency.json
