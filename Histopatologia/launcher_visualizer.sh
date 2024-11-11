#!/bin/bash 

#SBATCH --job-name="Visualizer"

#SBATCH -D .

#SBATCH --output=Visualizer_%j.out 

#SBATCH --error=Visualizer_%j.err 

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --time=01:00:00

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/bsc70642/containers/rocm_amd_t13_202310.sif bash -c "./run/visualize_samples.sh"
