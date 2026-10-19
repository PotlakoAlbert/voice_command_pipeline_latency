#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-fixthr
#SBATCH -o logs/mini-fixthr-%j.out
#SBATCH -t 00:10:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_fixed_thr.py
