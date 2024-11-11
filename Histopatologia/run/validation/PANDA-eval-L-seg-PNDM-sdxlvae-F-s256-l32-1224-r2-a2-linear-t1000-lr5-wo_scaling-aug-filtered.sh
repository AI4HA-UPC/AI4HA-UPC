XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --num_processes 1  evaluation.py --config configs/validation/PANDA-eval-L-seg-PNDM-sdxlvae-F-s256-l32-1224-r2-a2-linear-t1000-lr5-wo_scaling-aug-filtered.json