#!/bin/bash
#SBATCH -p batch
#SBATCH --exclude=mscluster129
#SBATCH -J mf-sweep
#SBATCH -o logs/mf-sweep-%A_%a.out
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-20
cd ~/voice_command_pipeline_latency
ARCHS=(crnn dilated_resnet repcnn)
BS=(0.0 0.1 0.2 0.33 0.5 1.0)
PY=~/conda_envs/kws-cu118/bin/python
if [ $SLURM_ARRAY_TASK_ID -lt 18 ]; then
  ARCH=${ARCHS[$((SLURM_ARRAY_TASK_ID / 6))]}
  B=${BS[$((SLURM_ARRAY_TASK_ID % 6))]}
  $PY src/mini_train_frame.py --arch $ARCH --b $B --seed ${SEED} --epochs 30 --patience 5
else
  ARCH=${ARCHS[$((SLURM_ARRAY_TASK_ID - 18))]}
  $PY src/mini_train_frame.py --arch $ARCH --b 0.0 --pooled-ce --seed ${SEED} --epochs 30 --patience 5 --out results/mini_frame_ce --ckpt-dir checkpoints/mini_frame_ce
fi
