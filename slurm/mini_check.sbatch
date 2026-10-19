#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-check
#SBATCH -o logs/mini-check-%j.out
#SBATCH -t 00:30:00
#SBATCH --cpus-per-task=4
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_check.py
