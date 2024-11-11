#!/bin/bash 

#SBATCH --job-name="PANDA-quality-data"

#SBATCH -D .

#SBATCH --output=PANDA-quality-data_%j.out 

#SBATCH --error=PANDA-quality-data_%j.err 

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --gres=gpu:1

#SBATCH --time=24:00:00

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/bsc70642/containers/rocm_amd_t13_202310.sif bash -c "./run/PANDA-quality-data.sh"
