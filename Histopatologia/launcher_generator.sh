#!/bin/bash 

#SBATCH --job-name="Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_soup_MN5-GenDataset"

#SBATCH -D .

#SBATCH --output=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_soup_MN5-GenDataset_%j.out 

#SBATCH --error=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_soup_MN5-GenDataset_%j.err 

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=64  

#SBATCH --gres=gpu:2

#SBATCH --time=48:00:00

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/MN4/bsc70/bsc70642/containers/rocm_amd_t13_202403.sif bash -c "./run/generation/Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_soup_MN5-GenDataset.sh"
