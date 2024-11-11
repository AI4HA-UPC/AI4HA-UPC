XDG_CACHE_HOME=/code
export XDG_CACHE_HOME


PYTHONPATH=/code
export PYTHONPATH   

wandb offline

WANDB_DIR=/code/logs
export WANDB_DIR

cd /code

PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512 accelerate launch --num_processes 1  ImageClassification-Inference.py --config configs/classification/PANDAGen-classification-ISUP-efficientnet-b4-256-lr4-cosine-Reinhard-Crossentropy-Model9.json