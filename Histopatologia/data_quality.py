import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from diffusion.data.ProstateDataPatches import ExamplesTrain, ExamplesVAE
from tqdm import tqdm 
import pandas as pd

# Define a function to process each batch
def process_batch(batch):
    batch_masks = batch['segmentation'][:, :, :, :].numpy()
    return batch_masks

# Function to calculate histogram for a given batch
def calculate_histogram(batch):
    num_samples, height, width, num_classes = batch.shape
    batch_reshaped = batch.reshape((num_samples * height * width, num_classes)) 
    #class_pixel = np.argmax(batch_reshaped, axis=1)
    batch_reshaped = torch.squeeze(batch_reshaped)
    hist, bins, _ = plt.hist(batch_reshaped, bins=range(0, max(batch_reshaped) + 2), edgecolor='black', alpha=0.7, density=True)
    return hist

# Function to plot histogram
"""
def plot_histogram(class_pixel):
    plt.figure(figsize=(16, 8))
    hist, bins, _ = plt.hist(class_pixel, bins=range(min(class_pixel), max(class_pixel) + 2), edgecolor='black', alpha=0.7, density=True)
    plt.title('Proportion of class values in all the pixels')
    plt.xlabel('Class value')
    plt.ylabel('Proportion')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, range(0, max(class_pixel)+1))
    plt.savefig('/gpfs/projects/bsc70/bsc70174/PANDA_code/logs/figures/class_distribution.png')
"""

def plot_histogram(histogram_values):
    plt.figure(figsize=(16, 8))
    
    bins = range(len(histogram_values) + 1)
    hist, bins, _ = plt.hist(range(len(histogram_values)), bins=bins, edgecolor='black', alpha=0.7, density=True, weights=histogram_values)
    
    plt.title('Proportion of class values in all the pixels')
    plt.xlabel('Class value')
    plt.ylabel('Proportion')
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    plt.xticks(bin_centers, range(0, len(histogram_values)))
    
    plt.savefig('/gpfs/projects/bsc70/bsc70174/PANDA_code/logs/figures/class_distribution.png')


# Load the dataset 
orig_data = ExamplesTrain(
    size=256, 
    data_csv="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/filtered_train.csv",
    data_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/patches",
    segmentation_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/masks",
    n_labels=6 
)
desired_dataset_size = np.round(len(orig_data) // 20)
random_indices = torch.randperm(len(orig_data))[:desired_dataset_size]
reduced_dataset = torch.utils.data.Subset(orig_data, random_indices)

# Set up your data and dataloader
nsamples = len(reduced_dataset)
mask_shape = reduced_dataset[0]['mask'].shape
print(mask_shape)
if torch.cuda.is_available():
    dataloader = DataLoader(reduced_dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=8)
else:
    dataloader = DataLoader(reduced_dataset, batch_size=2, shuffle=True)

# Iterate over the dataloader to get one-hot encoded masks and calculate histogram
final_hist = [0.0] * 6
i = 0
for batch in tqdm(dataloader):
    # Calculate histogram for the current batch
    hist_batch = calculate_histogram(batch['mask'])
    if len(hist_batch) == len(final_hist):
        final_hist = final_hist + hist_batch
    else:
        while len(hist_batch) < len(final_hist):
            hist_batch = np.append(hist_batch, [0.0])  
        final_hist = final_hist + hist_batch
    i += 1

# Plot the overall histogram
plot_histogram([x / i for x in final_hist])


# 
# Histogram of the Gleason Score 
# 
train_data_df = pd.read_csv("/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/filtered_train.csv")

gleason_values = np.unique(train_data_df['Gleason'])

plt.figure(figsize=(16, 8))
hist, bins, _ = plt.hist(train_data_df['Gleason'], bins=range(0, max(gleason_values)+2), edgecolor='black', alpha=0.7, density=True)
plt.title('Proportion of Gleason Score values in all the patches') 
plt.xlabel('Gleason value')
plt.ylabel('Proportion')
bin_centers = 0.5 * (bins[:-1] + bins[1:])  
plt.xticks(bin_centers, range(0, max(gleason_values)+1))
plt.savefig('/gpfs/projects/bsc70/bsc70174/PANDA_code/logs/figures/gleason_distribution.png')
