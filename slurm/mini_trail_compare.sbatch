#!/bin/bash
#SBATCH -p batch
#SBATCH -J mnt-compare
#SBATCH -o logs/mnt-compare-%j.out
#SBATCH -t 00:30:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_trail_compare.py
