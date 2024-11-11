XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --multi_gpu --num_processes 2  ImageClassification-Inference.py --config configs/classification/Prostate-classification-ISUP-efficientnet-b4-256-lr4-cosine-Reinhard-FocalLoss_MN5.json