#!/bin/bash

#BATCH --job-name=sjdf_data
#SBATCH --nodes=1 
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --output=_slurm_%j.out
#SBATCH --error=_slurm_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=john.doe@example.org

module purge

RUNDIR="$REPOPATH/MLmodels/data"

cd $RUNDIR

module load anaconda3/2020.07

conda create -n pytorch python=3.6
source activate pytorch
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install requests 
pip install matplotlib 
pip install sklearn 

python down_data.py &&
python proc_data.py 

