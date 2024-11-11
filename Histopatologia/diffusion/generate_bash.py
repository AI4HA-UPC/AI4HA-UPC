import os 
import subprocess 


def generate_bash(
    job_name,
    out_dir
):

    # Define the Bash script content with arguments
    bash_script_content = """
    #!/bin/bash

    #SBATCH --job-name="Prostate-LDM-s512-l64-att32-lpb2-scaled_linear-t1000-lr5"

    #SBATCH --qos=debug

    #SBATCH -D .

    #SBATCH --output=Prostate-LDM-s512-l64-att32-lpb2-scaled_linear-t1000-lr5_%j.out 

    #SBATCH --error=Prostate-LDM-s512-l64-att32-lpb2-scaled_linear-t1000-lr5_%j.err 

    #SBATCH --cpus-per-task=16

    #SBATCH --gres=gpu:1

    #SBATCH --time=02:00:00

    module purge
    module load rocm singularity

    export SINGULARITYENV_LD_LIBRARY_PATH=/opt/rocm/lib
    export SINGULARITYENV_PYTHONPATH=/root/.local/lib/python3.8/site-packages

    unset TMPDIR

    singularity  exec --rocm  --bind /gpfs/projects/bsc70/bsc70174/PANDA_resized_512/:/code --bind /gpfs/projects/bsc70/bsc70174/Data/:/data /gpfs/projects/bsc70/bsc70642/containers/rocm_amd_t13_202309.sif bash -c "./run/Prostate-LDM-s512-l64-att32-lpb2-scaled_linear-t1000-lr5.sh"
    
    """

# Define the name of the Bash script file
bash_script_filename = "my_script.sh"

# Write the Bash script content to a file
with open(bash_script_filename, "w") as bash_script_file:
    bash_script_file.write(bash_script_content)

# Make the Bash script executable
import os
os.chmod(bash_script_filename, 0o755)

# Execute the Bash script with arguments
import subprocess
subprocess.call(["./" + bash_script_filename, "Alice", "30"])
