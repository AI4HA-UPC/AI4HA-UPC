import numpy as np  
import matplotlib.pyplot as plt
import argparse
from accelerate import Accelerator
import torch
from tqdm.auto import tqdm
from diffusion.util import instantiate_from_config, load_config
from diffusion.data.ProstateDataPatches import ExamplesTrain
from diffusion.data.gen_datasets import ProstateGenDataset

from diffusion.validation.fls.fls.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from diffusion.validation.fls.fls.metrics.AuthPct import AuthPct
from diffusion.validation.fls.fls.metrics.FLS import FLS
from diffusion.validation.fls.fls.metrics.CTTest import CTTest
from diffusion.validation.fls.fls.metrics.FID import FID
from diffusion.validation.fls.fls.metrics.KID import KID
from diffusion.validation.fls.fls.metrics.PrecisionRecall import PrecisionRecall 


def main(config):
    accparams = config['accelerator']
    accelerator = Accelerator(**accparams)

    # Import the real and generated datasets
    train_data = instantiate_from_config(config['dataset']['train'])
    test_data = instantiate_from_config(config['dataset']['test'])
    generated_data = instantiate_from_config(config['dataset']['generated'])

    desired_dataset_size = len(train_data) // 2
    random_indices = torch.randperm(len(train_data))[:desired_dataset_size]
    reduced_dataset = torch.utils.data.Subset(train_data, random_indices)

    print("Train dataset size: ", len(reduced_dataset))
    print("Test dataset size: ", len(test_data))
    print("Generated dataset size: ", len(generated_data))
    
    # Feature extraction 
    print("Extracting features...")
    feature_extractor = DINOv2FeatureExtractor()

    train_feat = feature_extractor.get_features(reduced_dataset)
    test_feat = feature_extractor.get_features(test_data)
    gen_feat = feature_extractor.get_features(generated_data) 

    # FID 
    print("Computing FID...")
    fid_value = FID().compute_metric(train_feat, None, gen_feat)

    print(f"FID: {fid_value:.3f}")

    # FLS 
    print("Computing FLS...")
    fls_val = FLS().compute_metric(train_feat, test_feat, gen_feat)

    print(f"FLS: {fls_val:.3f}")

    # Ct Score 
    print("Computing Ct Score...")
    ct_value = CTTest().compute_metric(train_feat, test_feat, gen_feat) 

    print(f"Ct Score: {ct_value:.3f}")

    # KID 
    print("Computing KID...")
    kid_value = KID(ref_feat="test").compute_metric(None, test_feat, gen_feat)

    print(f"KID: {kid_value:.3f}")

    # Precision/Recall 
    print("Computing Precision/Recall...")
    precision_value = PrecisionRecall(mode="Precision").compute_metric(train_feat, None, gen_feat) # Default precision
    recall_value = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(train_feat, None, gen_feat) # Recall with k=5

    print(f"Precision: {precision_value:.3f}")
    print(f"Recall: {recall_value:.3f}")

    # Authenticity 
    print("Computing Authenticity...")
    auth_value = AuthPct().compute_metric(train_feat, test_feat, gen_feat)

    print(f"Authenticity: {auth_value:.3f}")

    # Return all metrics in a dictionary
    return {"FID": fid_value, "FLS": fls_val, "Ct Score": ct_value, "KID": kid_value,
            "Precision": precision_value, "Recall": recall_value, "Authenticity": auth_value}
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    main(config)