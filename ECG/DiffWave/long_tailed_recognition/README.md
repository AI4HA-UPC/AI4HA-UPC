# Long tailed recognition experiments
These experiments follow the framework presented in [1]:
1. First step: feature_extractor.py extracts features from a pretrained model on train data. Datasets used: ptbxl and chapman.
2. Second step: finetune_classifier.py takes the classifier head of the pretrained model and finetunes it with the extracted features and with synthetic generated features. Synthetic features are created using the extracted features and a DDIM model. The pretrained model is then updated with the finetuned classification head and tested on test data.