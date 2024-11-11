"""
Extract patches of the WSI images of the PANDA dataset. Only use the WSI images of the Radboud provider due to more detailed masks. 

This script stores patches and its corresponding mask in the output_dir and the size of the patches is controlled by patch_size. 
"""
import numpy as np
import pandas as pd
from PIL import Image
import zipfile
import albumentations
import openslide as slide
import os
from zipfile import ZipFile 
import csv

from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
import time, sys, warnings, glob
from tqdm import tqdm
from openslide.deepzoom import DeepZoomGenerator 
from joblib import Parallel, delayed

Image.MAX_IMAGE_PIXELS = None

labels = {0: 'background/unknown',
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

reverse_labels = {v: k for k, v in labels.items()}


def calculate_gleason_score(mask):
    # Step 1: Define the Gleason patterns
    patterns = [3, 4, 5] 

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
        return most_frequent_pattern + second_most_frequent_pattern
    else:
        return most_frequent_pattern * 2


# Function to generate the new gleason score
def gleason_score_new(mask):
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

        return reverse_labels[gleason_name]
    else:
        #most_frequent_pattern += 1
        gleason_name = f"{most_frequent_pattern}+{most_frequent_pattern}"
        
        return reverse_labels[gleason_name]


def thres_saturation(img, t1=15, t2=90):
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return (ave_sat >= t1) & (ave_sat <= t2)


def gleason_to_isup(gleason_score):
    if gleason_score == 0:
        return 0
    elif gleason_score == 1:
        return 1
    elif gleason_score == 2:
        return 2
    elif gleason_score == 3:
        return 3
    elif gleason_score == 4:
        return 4
    elif gleason_score == 5:
        return 5
    elif gleason_score <= 8:
        return 6
    else:
        return 7


def extract_patch(img, root_path, labels, patch_size, output_dir):
    image_ids = labels["image_ids"]

    # Import the WSI of the image and generate the patches 
    slide_image = slide.open_slide(os.path.join(root_path, 'train_images/'+image_ids[img]+'.tiff')) 
    tiles_image = DeepZoomGenerator(slide_image, tile_size=patch_size, overlap=0, limit_bounds=False)
    level_counts = tiles_image.level_count

    # Import the WSI of the mask and generate the patches 
    slide_mask = slide.open_slide(os.path.join(root_path, 'train_label_masks/'+image_ids[img]+'_mask.tiff'))
    tiles_mask = DeepZoomGenerator(slide_mask, tile_size=patch_size, overlap=0, limit_bounds=False)

    cols, rows = tiles_image.level_tiles[level_counts-1] 
    patches_name_list = []
    gleason_score_list = []
    isup_score_list = []
    for row in range(rows): #rows
        for col in range(cols): #cols
            tile_name_image = image_ids[img] + '_' + str(col) + "_" + str(row)
            tile_name_mask = image_ids[img] + '_' + str(col) + "_" + str(row) + '_mask'

            # Image 
            temp_tile_image = tiles_image.get_tile(level_counts-1, (col, row))
            temp_tile_RGB_image = temp_tile_image.convert('RGB')
            temp_tile_np_image = np.array(temp_tile_RGB_image)
            if not thres_saturation(temp_tile_np_image, t1=50, t2=90):
                continue
                            
            # Mask 
            temp_tile_mask = tiles_mask.get_tile(level_counts-1, (col, row))
            temp_tile_RGB_mask = temp_tile_mask.convert('RGB')
            temp_tile_np_mask = np.array(temp_tile_RGB_mask)
            temp_tile_np_mask = temp_tile_np_mask[..., 0]

            gleason_score = gleason_score_new(temp_tile_np_mask) 
            gleason_score_list.append(gleason_score)
            patches_name_list.append(tile_name_image)
            isup_score_list.append(gleason_to_isup(gleason_score))
            
            # Save patches 
            final_patch = Image.fromarray(temp_tile_np_image)
            final_patch.save(os.path.join(output_dir, 'patches', tile_name_image + ".png"))
            final_mask = Image.fromarray(temp_tile_np_mask)
            final_mask.save(os.path.join(output_dir, 'masks', tile_name_mask + ".png"))

    return patches_name_list, gleason_score_list, isup_score_list


def main():
    root_path = '/gpfs/projects/bsc70/bsc70174/Data/PANDA' 
    output_dir = '/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_512_v2' 
    patch_size = 512
    path_csv = 'train_reduced.csv'

    image_ids = []
    gleason_score = []
    with open(os.path.join(root_path, path_csv), "r") as f:
        lines = f.read().splitlines()
        for line in lines[1:]:
            if line.split(',')[2] == 'radboud':
                image_ids.append(line.split(',')[1])

    labels = {
        "image_ids": image_ids,
        "relative_file_path": [l+'.tiff' for l in image_ids],
        "file_path": [os.path.join(root_path, 'train_images/'+l+'.tiff') for l in image_ids],
        "segmentation_path_": [os.path.join(root_path, 'train_label_masks/'+l+'_mask.tiff') for l in image_ids]
    }

    # Define the output structure
    os.makedirs(output_dir, exist_ok=True)

    patches_path = os.path.join(output_dir, 'patches')
    os.makedirs(patches_path, exist_ok=True)

    masks_path = os.path.join(output_dir, 'masks')
    os.makedirs(masks_path, exist_ok=True)

    # Use joblib to parallelize patch extraction
    num_jobs = 64
    args_list = [(img, root_path, labels, patch_size, output_dir) for img in range(len(labels["file_path"]))] 
    results = Parallel(n_jobs=num_jobs)(delayed(extract_patch)(*args) for args in tqdm(args_list, desc="Processing Images"))

    patches_name_list = []
    gleason_score_list = []
    isup_score_list = []
    # Collect results from parallel processes
    for result in results:
        patches_name_list.extend(result[0])
        gleason_score_list.extend(result[1])
        isup_score_list.extend(result[2])

        labels_dict = {'Image_id': patches_name_list, 'Gleason': gleason_score_list, 'ISUP': isup_score_list}
        labels_df = pd.DataFrame.from_dict(labels_dict) 
        labels_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)


if __name__ == "__main__":
    main()
