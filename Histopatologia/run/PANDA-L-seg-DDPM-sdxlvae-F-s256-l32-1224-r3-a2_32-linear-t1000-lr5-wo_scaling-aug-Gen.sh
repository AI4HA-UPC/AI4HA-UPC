XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --num_processes 1  DiffusionGenerator.py --config configs/PANDA-L-seg-DDPM-sdxlvae-F-s256-l32-1224-r3-a2_32-linear-t1000-lr5-wo_scaling-aug.json