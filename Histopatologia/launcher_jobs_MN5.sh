#!/bin/bash

# Define the first job (Train 1)
first_job=$(sbatch <<EOF
#!/bin/bash

#SBATCH --job-name="Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5"
#SBATCH -D .
#SBATCH --output=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5_%j.out 
#SBATCH --error=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5_%j.err 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/MN4/bsc70/bsc70642/containers/rocm_amd_t13_202403.sif bash -c "./run/Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5.sh"
EOF
)

# Extract the job ID of the first job (Train 2)
first_job_id=$(echo $first_job | awk '{print $4}')

# Define the second job
second_job=$(sbatch <<EOF
#!/bin/bash

#SBATCH --job-name="Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5"
#SBATCH -D .
#SBATCH --output=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5_%j.out 
#SBATCH --error=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5_%j.err 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --dependency=afterok:$first_job_id

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/MN4/bsc70/bsc70642/containers/rocm_amd_t13_202403.sif bash -c "./run/Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5.sh"
EOF
)

# Extract the job ID of the second job 
second_job_id=$(echo $second_job | awk '{print $4}')

# Define the third job (Train 3)
third_job=$(sbatch <<EOF
#!/bin/bash

#SBATCH --job-name="Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5"
#SBATCH -D .
#SBATCH --output=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5_%j.out 
#SBATCH --error=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5_%j.err 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --dependency=afterok:$second_job_id

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/MN4/bsc70/bsc70642/containers/rocm_amd_t13_202403.sif bash -c "./run/Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5.sh"
EOF
)

# Extract the job ID of the third job 
third_job_id=$(echo $third_job | awk '{print $4}')

# Define the fourth job (Generation)
fourth_job=$(sbatch <<EOF
#!/bin/bash

#SBATCH --job-name="Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5-Gen"
#SBATCH -D .
#SBATCH --output=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5-Gen_%j.out 
#SBATCH --error=Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5-Gen_%j.err 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --dependency=afterok:$third_job_id

module purge
module load rocm singularity

export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

unset TMPDIR

singularity  exec --rocm  --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/:/code --bind /gpfs/projects/bsc70/MN4/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/MN4/bsc70/bsc70642/containers/rocm_amd_t13_202403.sif bash -c "./run/Prostate-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5-Inference.sh"
EOF
)