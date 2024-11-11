import numpy as np
import os 
from PIL import Image
import pandas as pd
from tqdm import tqdm 



def calculate_gleason_score(mask):
    # Step 1: Define the Gleason patterns
    patterns = [2, 3, 4] 

    # Step 2: Compute the area occupied by each Gleason pattern
    area_percentage = {}
    total_area = mask.size
    for pattern in patterns:
        area_percentage[pattern] = (mask == pattern).sum() / total_area
    
    if sum(area_percentage.values()) == 0:
        return 0

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
        most_frequent_pattern += 1
        second_most_frequent_pattern += 1
        return most_frequent_pattern + second_most_frequent_pattern
    else:
        most_frequent_pattern += 1
        return most_frequent_pattern * 2


root_path = '/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_512' 
patch_size = 512
original_csv = 'train.csv'
new_csv = 'simple_train_radboud.csv' 

image_ids = []
gleason_score_list = []
with open(os.path.join(root_path, original_csv), "r") as f:
    lines = f.read().splitlines()
    for line in lines[1:]:
        if line.split(',')[6] == 'radboud':
            image_ids.append(line.split(',')[1])

for id in tqdm(image_ids):
    mask = Image.open(os.path.join(root_path, 'masks/'+id+'_mask.png')) 
    mask = np.array(mask).astype(np.uint8)

    gleason_score = calculate_gleason_score(mask) 
    gleason_score_list.append(gleason_score)

final_dict = {
    "Image_id": image_ids,
    "Gleason": gleason_score_list
}
final_df = pd.DataFrame.from_dict(final_dict) 
final_df.to_csv(os.path.join(root_path, new_csv))
