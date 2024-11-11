import numpy as np
import cv2
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import pandas as pd


# Function to process a single image-mask pair
def process_image_mask(image_path, mask_path, gleason, threshold):
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # Count the number of pixels with class 1 in the mask
    class_1_pixels = np.count_nonzero(mask == 1)
    class_higher_2_pixels = np.count_nonzero(mask > 2)

    # Calculate the total number of pixels in the mask
    total_pixels = mask.size

    # Calculate the ratio of class 1 pixels to total pixels
    class_1_ratio = class_1_pixels / total_pixels
    class_higher_2_ration = class_higher_2_pixels / total_pixels

    # Check if the imbalance ratio is below the threshold
    if class_higher_2_ration > 0 and class_1_ratio <= threshold:
        #fig, axs = plt.subplots(1, 2)
        #axs[0].imshow(image)
        #axs[1].imshow(mask/5, cmap='gray')
        #plt.show()
        return image_path, mask_path, gleason
    else:
        #fig, axs = plt.subplots(1, 2)
        #axs[0].imshow(image)
        #axs[1].imshow(mask/5, cmap='gray')
        #plt.show()
        return None, None, None

# Function to filter imbalanced image-mask pairs using joblib
def filter_imbalance_parallel(image_paths, mask_paths, gleason_list, threshold, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_image_mask)(image_path, mask_path, gleason, threshold)
        for image_path, mask_path, gleason in zip(image_paths, mask_paths, gleason_list)
    )

    # Filter out None results
    filtered_results = [(img, msk, gleason) for img, msk, gleason in results if img is not None and msk is not None]

    return zip(*filtered_results)


# Example usage 
csv_path = '/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/train.csv'
data_path = '/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/'
image_ids = []
gleason_list = []
with open(csv_path, "r") as f:
    lines = f.read().splitlines()
    for line in lines[1:]:
        image_ids.append(line.split(',')[1])
        gleason_list.append(line.split(',')[2])

image_paths = [os.path.join(data_path+'patches/', l+'.png') for l in image_ids] 
mask_paths = [os.path.join(data_path+'masks/', l+'_mask.png') for l in image_ids] 

print("Number of images: ", len(image_paths))
print("Number of masks: ", len(mask_paths))

threshold = 0.6  
n_jobs = 64 

filtered_images, filtered_masks, filtered_gleason = filter_imbalance_parallel(image_paths, mask_paths, gleason_list, threshold, n_jobs)

filtered_images = [p.split('/')[-1].split('.')[0] for p in filtered_images]
filtered_masks = [p.split('/')[-1].split('.')[0] for p in filtered_masks]

print("Number of images: ", len(filtered_images))
print("Number of masks: ", len(filtered_masks))

labels_dict = {'Image_id': filtered_images, 'Gleason': filtered_gleason}
labels_df = pd.DataFrame.from_dict(labels_dict) 
labels_df.to_csv(os.path.join(data_path, "filtered_train.csv"))