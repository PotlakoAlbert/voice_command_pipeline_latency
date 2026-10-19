#!/bin/bash
#SBATCH -p batch
#SBATCH -J mini-multiseed
#SBATCH -o logs/mini-multiseed-%j.out
#SBATCH -t 00:30:00
cd ~/voice_command_pipeline_latency
~/conda_envs/kws-cu118/bin/python src/mini_multiseed.py
