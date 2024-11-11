"""
Generate new gleason value for each pair of image and mask
"""
import os
import numpy as np
import pandas as pd
import csv
import torch 
from tqdm import tqdm   
from joblib import Parallel, delayed    
from torch.utils.data import DataLoader

from diffusion.data.ProstateDataPatches import ExamplesTrain


# Function to generate the new gleason score
def gleason_score_new(mask):

    mask = mask.numpy()
    
    # Step 1: Define the Gleason patterns
    patterns = [3, 4, 5] 
    values = [0, 1, 2]

    # Step 2: Compute the area occupied by each Gleason pattern
    area_percentage = {}
    total_area = mask.size
    for pattern in patterns:
        area_percentage[pattern] = (mask == pattern).sum() / total_area
    
    if sum(area_percentage.values()) == 0: # No gleason pattern detected
        # Determine the most frequent value
        area_percentage = {}
        total_area = mask.size
        for pattern in values:
            area_percentage[pattern] = (mask == pattern).sum() / total_area
        most_frequent_value = max(area_percentage, key=area_percentage.get)

        return most_frequent_value
    
    # Step 3: Determine the most frequent pattern
    most_frequent_pattern = max(area_percentage, key=area_percentage.get)

    # Step 4: Extract the highest Gleason value present in the mask 
    highest_pattern_present = []
    for pattern, percentage in area_percentage.items():
        if percentage > 0:
            highest_pattern_present.append(pattern)
    highest_pattern_present = max(highest_pattern_present)

    # Step 5: Determine the second most frequent pattern
    second_most_frequent_pattern = None
    second_most_frequent_percentage = 0
    for pattern in patterns:
        if pattern != most_frequent_pattern:
            if (area_percentage[pattern] >= 0.05 and area_percentage[pattern] > second_most_frequent_percentage) or pattern == highest_pattern_present:
                second_most_frequent_pattern = pattern
                second_most_frequent_percentage = area_percentage[pattern]

    # Step 6: Calculate the Gleason score
    if second_most_frequent_pattern is not None:
        #most_frequent_pattern += 1
        #second_most_frequent_pattern += 1
        gleason_name = f"{most_frequent_pattern}+{second_most_frequent_pattern}"

        return reverse_new_labels[gleason_name]
    else:
        #most_frequent_pattern += 1
        gleason_name = f"{most_frequent_pattern}+{most_frequent_pattern}"
        
        return reverse_new_labels[gleason_name]


if __name__ == "__main__":

    # Load the data
    orig_data = ExamplesTrain(
        size=256, 
        data_csv="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/train.csv",
        data_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/patches",
        segmentation_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/masks",
        n_labels=6 
    )

    # Define the new labels
    new_labels = {0: 'background/unknown',
                  1: 'stroma',
                  2: 'healthy epithelium',
                  3: '3+3',
                  4: '3+4',
                  5: '4+3',
                  6: '4+4',
                  7: '3+5',
                  8: '5+3',
                  9: '4+5',
                  10: '5+4',
                  11: '5+5'}

    reverse_new_labels = {v: k for k, v in new_labels.items()}

    # Define the batch size
    batch_size = 128  # Adjust batch size based on your memory capacity

    # Create the dataloader
    dataloader = DataLoader(orig_data, batch_size=batch_size)

    # Generate the new gleason score for each pair of image and mask
    results = []
    for ex in tqdm(dataloader):
        scores = Parallel(n_jobs=-1)(delayed(gleason_score_new)(m) for m in ex['mask'])
        for i, score in enumerate(scores):
            results.append([ex['image_name'][i], score])

    # Save the results to a CSV file
    with open('/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/new_train.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File', 'Score'])
        for row in results:
            writer.writerow(row)