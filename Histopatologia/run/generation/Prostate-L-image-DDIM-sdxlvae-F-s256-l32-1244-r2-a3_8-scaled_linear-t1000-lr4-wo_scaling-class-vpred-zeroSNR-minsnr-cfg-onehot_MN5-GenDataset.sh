XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --num_processes 1  DiffusionGeneratorDataset.py --config configs/generation/Prostate-L-image-DDIM-sdxlvae-F-s256-l32-1244-r2-a3_8-scaled_linear-t1000-lr4-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot_MN5-GenDataset.json