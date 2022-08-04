#!/bin/bash
#SBATCH -J Noisetunnel
#SBATCH -o BestRegion_20samples.txt
#SBATCH -e BestRegion_20samplesError.txt
#SBATCH --nodes=1
#SBATCH -c 10
#SBATCH --mem 100G
#SBATCH --gres=gpu:v100:2
#SBATCH -p qTRDGPUH,qTRDGPUM,qTRDGPUL
#SBATCH --oversubscribe
#SBATCH -t 7200

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/vjoshi6/bin/miniconda3/lib
python3 /data/users2/vjoshi6/bin/pythonFiles/CatalystIntrospection/Catalyst_complex/Noisetunner.py
