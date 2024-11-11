XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --multi_gpu --num_processes 2  DiffusionConditionalLatentPreTrain.py --config configs/PANDA-clean-L-image-DPMSolver-sdxlvae-F-s256-l32-1244-r2-a5_10_20-scaled_linear-t1000-lr5-wo_scaling-class-vpred-zeroSNR-minsnr-cfg-onehot.json