XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --num_processes 1  GenerateLatentsV2.py --config configs/PANDA-L-s512-l32-sdxl-vae-aug.json