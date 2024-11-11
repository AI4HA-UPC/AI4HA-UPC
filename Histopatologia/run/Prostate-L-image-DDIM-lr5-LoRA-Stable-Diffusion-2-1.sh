XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code:/usr/lib/python3.8/site-packages
export PYTHONPATH

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --multi_gpu --num_processes 2  DiffusionLoRA.py --config configs/Prostate-L-image-DDIM-lr5-LoRA-Stable-Diffusion-2-1.json