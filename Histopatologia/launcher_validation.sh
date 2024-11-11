#!/bin/bash 

#SBATCH --job-name="Validation-PANDA-L-seg-DDPM-DDIM-sdxlvae-F-s256-l32-1244-r2-a3_8-scaled_linear-t1000-lr4-wo_scaling-aug-class-SNR"

#SBATCH -D .

#SBATCH --output=Validation-PANDA-L-seg-DDPM-DDIM-sdxlvae-F-s256-l32-1244-r2-a3_8-scaled_linear-t1000-lr4-wo_scaling-aug-class-SNR_%j.out 

#SBATCH --error=Validation-PANDA-L-seg-DDPM-DDIM-sdxlvae-F-s256-l32-1244-r2-a3_8-scaled_linear-t1000-lr4-wo_scaling-aug-class-SNR_%j.err 

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=32

#SBATCH --gres=gpu:1

#SBATCH --time=10:00:00

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/bsc70642/containers/rocm_amd_t13_202312.sif bash -c "./run/validation/PANDA-eval-L-seg-DDPM-DDIM-sdxlvae-F-s256-l32-1244-r2-a3_8-scaled_linear-t1000-lr4-wo_scaling-aug-class-SNR.sh"
