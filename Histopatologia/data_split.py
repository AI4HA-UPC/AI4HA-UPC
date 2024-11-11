"""
Script to split the dataset into train and validation
""" 
import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_512_v2/train_combined.csv')

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the split information to CSV files
train_df.to_csv('/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_512_v2/train_split_combined.csv', index=False)
val_df.to_csv('/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_512_v2/val_split_combined.csv', index=False)

