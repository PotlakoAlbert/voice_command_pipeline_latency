#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-cache
#SBATCH -o logs/mini-cache-%j.out
#SBATCH -t 02:00:00
#SBATCH --cpus-per-task=4
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_cache.py
