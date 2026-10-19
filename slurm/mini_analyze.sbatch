#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-analyze
#SBATCH -o logs/mini-analyze-%j.out
#SBATCH -t 00:20:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_analyze.py
