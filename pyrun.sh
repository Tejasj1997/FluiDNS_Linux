#!/bin/bash
# SBATCH --export=ALL
# SBATCH --partition=iist-all
# SBATCH --mail-type=ALL
# SBATCH --mail-user=jeurkar.sc19m016@pg.iist.ac.in
# SBATCH --job-name=test_job
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=2
pwd; hostname; date
source ~/Desktop/local_py/bin/activate
echo starting...
python ~/ins_ibm_study/FluiDNS.py
echo Job completed
