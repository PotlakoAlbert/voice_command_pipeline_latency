#!/bin/bash
#SBATCH -p batch
#SBATCH -J asr_nokws
#SBATCH -o logs/asr_nokws-%A_%a.out
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/asr_nokws.py $SLURM_ARRAY_TASK_ID 12
